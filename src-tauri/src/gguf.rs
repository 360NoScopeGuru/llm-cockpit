//! Minimal GGUF metadata parser.
//!
//! Reads only the header + metadata key/value block of a GGUF file — enough to
//! surface a model's architecture, quantization, context length and layout in the
//! library view — without ever loading tensor weights. Large arrays (e.g. the
//! tokenizer vocab) are skipped in-place rather than allocated.
//!
//! Reference: GGUF spec (v2/v3), little-endian.
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::Serialize;
use thiserror::Error;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0} (only v2/v3 supported)")]
    UnsupportedVersion(u32),
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),
    #[error("nested arrays are not supported by the GGUF spec")]
    NestedArray,
    #[error("malformed metadata: {0}")]
    Malformed(String),
}

/// A parsed GGUF metadata value. Arrays are not materialized — we keep only
/// their element type and length so huge vocab arrays cost nothing.
///
/// The full scalar set is retained for fidelity / future power-user KV
/// inspection even though the cockpit only reads a few of them today.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array { elem_type: u32, len: u64 },
}

impl GgufValue {
    /// Coerce any integer-ish scalar to u64 (used for counts/lengths).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// The subset of GGUF metadata the cockpit cares about, plus the raw KV map for
/// power-user inspection later.
#[derive(Debug, Clone, Serialize)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub architecture: Option<String>,
    pub name: Option<String>,
    /// Human-readable quant label derived from `general.file_type` (e.g. "Q4_K_M").
    pub quant_label: Option<String>,
    pub file_type: Option<u32>,
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub key_length: Option<u64>,
    pub value_length: Option<u64>,
    pub parameter_count: Option<u64>,
    pub size_label: Option<String>,
    /// For sharded models: total shard count and this file's 1-based index.
    pub split_count: Option<u64>,
    pub split_no: Option<u64>,
}

/// GGUF metadata value type tags.
mod vtype {
    pub const U8: u32 = 0;
    pub const I8: u32 = 1;
    pub const U16: u32 = 2;
    pub const I16: u32 = 3;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const U64: u32 = 10;
    pub const I64: u32 = 11;
    pub const F64: u32 = 12;
}

/// Fixed byte size for scalar element types, used to skip numeric arrays by seek.
fn scalar_size(elem_type: u32) -> Option<u64> {
    match elem_type {
        vtype::U8 | vtype::I8 | vtype::BOOL => Some(1),
        vtype::U16 | vtype::I16 => Some(2),
        vtype::U32 | vtype::I32 | vtype::F32 => Some(4),
        vtype::U64 | vtype::I64 | vtype::F64 => Some(8),
        _ => None,
    }
}

struct Reader<R: Read + Seek> {
    inner: R,
}

impl<R: Read + Seek> Reader<R> {
    fn new(inner: R) -> Self {
        Reader { inner }
    }

    fn read_exact_n<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut buf = [0u8; N];
        self.inner.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_exact_n::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.read_exact_n::<2>()?))
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.read_exact_n::<4>()?))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.read_exact_n::<8>()?))
    }
    fn i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.u8()? as i8)
    }
    fn i16(&mut self) -> Result<i16, GgufError> {
        Ok(i16::from_le_bytes(self.read_exact_n::<2>()?))
    }
    fn i32(&mut self) -> Result<i32, GgufError> {
        Ok(i32::from_le_bytes(self.read_exact_n::<4>()?))
    }
    fn i64(&mut self) -> Result<i64, GgufError> {
        Ok(i64::from_le_bytes(self.read_exact_n::<8>()?))
    }
    fn f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_le_bytes(self.read_exact_n::<4>()?))
    }
    fn f64(&mut self) -> Result<f64, GgufError> {
        Ok(f64::from_le_bytes(self.read_exact_n::<8>()?))
    }

    /// A GGUF string: u64 length prefix followed by that many UTF-8 bytes.
    /// Guards against absurd lengths so a corrupt/misaligned file can't try to
    /// allocate gigabytes.
    fn gguf_string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > 64 * 1024 * 1024 {
            return Err(GgufError::Malformed(format!(
                "implausible string length {len}"
            )));
        }
        let mut buf = vec![0u8; len as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip(&mut self, n: u64) -> Result<(), GgufError> {
        self.inner.seek(SeekFrom::Current(n as i64))?;
        Ok(())
    }

    /// Read a single metadata value of the given type, skipping array bodies.
    fn value(&mut self, vt: u32) -> Result<GgufValue, GgufError> {
        Ok(match vt {
            vtype::U8 => GgufValue::U8(self.u8()?),
            vtype::I8 => GgufValue::I8(self.i8()?),
            vtype::U16 => GgufValue::U16(self.u16()?),
            vtype::I16 => GgufValue::I16(self.i16()?),
            vtype::U32 => GgufValue::U32(self.u32()?),
            vtype::I32 => GgufValue::I32(self.i32()?),
            vtype::F32 => GgufValue::F32(self.f32()?),
            vtype::BOOL => GgufValue::Bool(self.u8()? != 0),
            vtype::STRING => GgufValue::Str(self.gguf_string()?),
            vtype::U64 => GgufValue::U64(self.u64()?),
            vtype::I64 => GgufValue::I64(self.i64()?),
            vtype::F64 => GgufValue::F64(self.f64()?),
            vtype::ARRAY => {
                let elem_type = self.u32()?;
                let len = self.u64()?;
                self.skip_array_body(elem_type, len)?;
                GgufValue::Array { elem_type, len }
            }
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    /// Advance past an array's elements without allocating them.
    fn skip_array_body(&mut self, elem_type: u32, len: u64) -> Result<(), GgufError> {
        if let Some(size) = scalar_size(elem_type) {
            self.skip(size.saturating_mul(len))?;
        } else if elem_type == vtype::STRING {
            // Variable-length: must read each length prefix, but never allocate the bytes.
            for _ in 0..len {
                let slen = self.u64()?;
                self.skip(slen)?;
            }
        } else if elem_type == vtype::ARRAY {
            return Err(GgufError::NestedArray);
        } else {
            return Err(GgufError::UnknownValueType(elem_type));
        }
        Ok(())
    }
}

/// Parse the header + metadata KV block of a GGUF file at `path`.
pub fn read_gguf_metadata(path: &Path) -> Result<GgufMetadata, GgufError> {
    let file = File::open(path)?;
    read_metadata(BufReader::new(file))
}

/// Parse GGUF metadata from any seekable reader. Split out from the file path
/// entry point so it can be exercised against in-memory buffers in tests.
pub fn read_metadata<R: Read + Seek>(reader: R) -> Result<GgufMetadata, GgufError> {
    let mut r = Reader::new(reader);

    let magic = r.read_exact_n::<4>()?;
    if &magic != GGUF_MAGIC {
        return Err(GgufError::BadMagic);
    }

    let version = r.u32()?;
    if version != 2 && version != 3 {
        return Err(GgufError::UnsupportedVersion(version));
    }

    let tensor_count = r.u64()?;
    let metadata_kv_count = r.u64()?;

    let mut kv: HashMap<String, GgufValue> = HashMap::new();
    for _ in 0..metadata_kv_count {
        let key = r.gguf_string()?;
        let vt = r.u32()?;
        let value = r.value(vt)?;
        kv.insert(key, value);
    }

    let architecture = kv
        .get("general.architecture")
        .and_then(|v| v.as_string())
        .map(str::to_owned);

    // Architecture-scoped keys, e.g. "llama.context_length".
    let arch_key = |suffix: &str| -> Option<u64> {
        let arch = architecture.as_deref()?;
        kv.get(&format!("{arch}.{suffix}")).and_then(|v| v.as_u64())
    };

    let file_type = kv
        .get("general.file_type")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    Ok(GgufMetadata {
        version,
        tensor_count,
        metadata_kv_count,
        name: kv
            .get("general.name")
            .and_then(|v| v.as_string())
            .map(str::to_owned),
        quant_label: file_type.map(file_type_label),
        file_type,
        context_length: arch_key("context_length"),
        block_count: arch_key("block_count"),
        embedding_length: arch_key("embedding_length"),
        head_count: arch_key("attention.head_count"),
        head_count_kv: arch_key("attention.head_count_kv"),
        key_length: arch_key("attention.key_length"),
        value_length: arch_key("attention.value_length"),
        parameter_count: kv.get("general.parameter_count").and_then(|v| v.as_u64()),
        size_label: kv
            .get("general.size_label")
            .and_then(|v| v.as_string())
            .map(str::to_owned),
        split_count: kv.get("split.count").and_then(|v| v.as_u64()),
        split_no: kv.get("split.no").and_then(|v| v.as_u64()),
        architecture,
    })
}

/// Map `general.file_type` (llama.cpp LLAMA_FTYPE enum) to a human quant label.
/// Unknown values fall back to `unknown(<n>)`.
pub fn file_type_label(ft: u32) -> String {
    let s = match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        33 => "TQ1_0",
        34 => "TQ2_0",
        other => return format!("unknown({other})"),
    };
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Helpers to build a synthetic little-endian GGUF byte buffer.
    fn push_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend((s.len() as u64).to_le_bytes());
        buf.extend(s.as_bytes());
    }
    fn kv_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        push_str(buf, key);
        buf.extend(vtype::STRING.to_le_bytes());
        push_str(buf, val);
    }
    fn kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
        push_str(buf, key);
        buf.extend(vtype::U32.to_le_bytes());
        buf.extend(v.to_le_bytes());
    }
    /// A string-array KV (like the tokenizer vocab) that the parser must skip
    /// over without materializing, while still reading keys that follow it.
    fn kv_str_array(buf: &mut Vec<u8>, key: &str, vals: &[&str]) {
        push_str(buf, key);
        buf.extend(vtype::ARRAY.to_le_bytes());
        buf.extend(vtype::STRING.to_le_bytes());
        buf.extend((vals.len() as u64).to_le_bytes());
        for v in vals {
            push_str(buf, v);
        }
    }

    fn build_gguf(kv_count: u64, body: Vec<u8>) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend(GGUF_MAGIC);
        buf.extend(3u32.to_le_bytes()); // version
        buf.extend(0u64.to_le_bytes()); // tensor_count
        buf.extend(kv_count.to_le_bytes());
        buf.extend(body);
        buf
    }

    #[test]
    fn parses_synthetic_gguf() {
        let mut body = Vec::new();
        kv_str(&mut body, "general.architecture", "llama");
        kv_str(&mut body, "general.name", "Test Model");
        // Big array in the middle: parser must skip it and still read what follows.
        kv_str_array(&mut body, "tokenizer.ggml.tokens", &["a", "bb", "ccc"]);
        kv_u32(&mut body, "general.file_type", 15); // Q4_K_M
        kv_u32(&mut body, "llama.context_length", 4096);
        kv_u32(&mut body, "llama.block_count", 32);

        let bytes = build_gguf(6, body);
        let md = read_metadata(Cursor::new(bytes)).expect("should parse");

        assert_eq!(md.version, 3);
        assert_eq!(md.architecture.as_deref(), Some("llama"));
        assert_eq!(md.name.as_deref(), Some("Test Model"));
        assert_eq!(md.quant_label.as_deref(), Some("Q4_K_M"));
        assert_eq!(md.context_length, Some(4096));
        assert_eq!(md.block_count, Some(32));
    }

    #[test]
    fn rejects_bad_magic() {
        let bytes = vec![b'N', b'O', b'P', b'E', 0, 0, 0, 0];
        assert!(matches!(
            read_metadata(Cursor::new(bytes)),
            Err(GgufError::BadMagic)
        ));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut buf = Vec::new();
        buf.extend(GGUF_MAGIC);
        buf.extend(1u32.to_le_bytes()); // v1 unsupported
        buf.extend(0u64.to_le_bytes());
        buf.extend(0u64.to_le_bytes());
        assert!(matches!(
            read_metadata(Cursor::new(buf)),
            Err(GgufError::UnsupportedVersion(1))
        ));
    }
}
