//! Versioned, bounded, checksummed CPU f64 model/Adam checkpoints.
//! The checksum detects accidental corruption; it is not authentication.
use crate::{
    optimizer::{AdamConfig, Moments},
    tensor::{self, Tensor},
};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};
const MAGIC: &[u8; 8] = b"MLCKPT01";
const MAX_FILE: usize = 128 * 1024 * 1024 + 1024 * 1024;
const MAX_PARAMETERS: usize = 4096;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Checkpoint {
    pub parameters: BTreeMap<String, Tensor>,
    pub adam: BTreeMap<String, Moments>,
    pub adam_config: Option<AdamConfig>,
}
fn valid_name(n: &str) -> bool {
    let mut c = n.chars();
    n.len() <= 128
        && c.next()
            .is_some_and(|x| x.is_ascii_alphabetic() || x == '_')
        && c.all(|x| x.is_ascii_alphanumeric() || x == '_')
}
impl Checkpoint {
    pub fn validate(&self) -> Result<(), String> {
        if self.parameters.len() > MAX_PARAMETERS {
            return Err("too many checkpoint parameters".into());
        }
        if let Some(config) = self.adam_config {
            config.validate()?;
        }
        if !self.adam.is_empty() && self.adam_config.is_none() {
            return Err("checkpoint Adam states require configuration".into());
        }
        let mut total = 0usize;
        for (name, t) in &self.parameters {
            if !valid_name(name) || t.shape.len() > 8 {
                return Err("invalid checkpoint parameter name or rank".into());
            }
            if tensor::elements(&t.shape)? != t.data.len() || t.data.iter().any(|x| !x.is_finite())
            {
                return Err("invalid checkpoint tensor".into());
            }
            total = total
                .checked_add(t.data.len())
                .ok_or("checkpoint size overflow")?;
            if let Some(state) = self.adam.get(name) {
                if state.step == 0
                    || state.first.len() != t.data.len()
                    || state.second.len() != t.data.len()
                    || state.first.iter().any(|x| !x.is_finite())
                    || state.second.iter().any(|x| !x.is_finite() || *x < 0.0)
                {
                    return Err("invalid checkpoint Adam moments".into());
                }
                total = total
                    .checked_add(
                        t.data
                            .len()
                            .checked_mul(2)
                            .ok_or("checkpoint size overflow")?,
                    )
                    .ok_or("checkpoint size overflow")?;
            }
        }
        if total > tensor::MAX_ELEMENTS
            || self.adam.keys().any(|n| !self.parameters.contains_key(n))
        {
            return Err("checkpoint exceeds storage limit or contains unknown Adam states".into());
        }
        Ok(())
    }
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        self.validate()?;
        let mut out = MAGIC.to_vec();
        out.push(u8::from(self.adam_config.is_some()));
        if let Some(c) = self.adam_config {
            for x in [c.beta1, c.beta2, c.epsilon] {
                out.extend(x.to_le_bytes());
            }
        }
        out.extend((self.parameters.len() as u32).to_le_bytes());
        for (name, t) in &self.parameters {
            out.extend((name.len() as u32).to_le_bytes());
            out.extend(name.as_bytes());
            out.extend((t.shape.len() as u32).to_le_bytes());
            for &d in &t.shape {
                out.extend((d as u64).to_le_bytes());
            }
            out.extend((t.data.len() as u64).to_le_bytes());
            for &x in &t.data {
                out.extend(x.to_le_bytes());
            }
            out.push(u8::from(self.adam.contains_key(name)));
            if let Some(s) = self.adam.get(name) {
                out.extend(s.step.to_le_bytes());
                for &x in s.first.iter().chain(&s.second) {
                    out.extend(x.to_le_bytes());
                }
            }
        }
        let checksum = checksum(&out);
        out.extend(checksum.to_le_bytes());
        if out.len() > MAX_FILE {
            return Err("checkpoint file exceeds size limit".into());
        }
        Ok(out)
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 21 || bytes.len() > MAX_FILE {
            return Err("invalid checkpoint file size".into());
        }
        if &bytes[..8] != MAGIC {
            return Err("unsupported checkpoint format/version".into());
        }
        let payload = &bytes[..bytes.len() - 8];
        let stored = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap());
        if checksum(payload) != stored {
            return Err("checkpoint checksum mismatch".into());
        }
        let mut reader = Reader {
            bytes: payload,
            pos: 8,
        };
        let adam_config = if reader.flag()? {
            let c = AdamConfig {
                beta1: reader.float()?,
                beta2: reader.float()?,
                epsilon: reader.float()?,
            };
            c.validate()?;
            Some(c)
        } else {
            None
        };
        let count = reader.u32()? as usize;
        if count > MAX_PARAMETERS {
            return Err("too many checkpoint parameters".into());
        }
        let mut checkpoint = Self {
            adam_config,
            ..Self::default()
        };
        let mut total = 0usize;
        for _ in 0..count {
            let len = reader.u32()? as usize;
            if len > 128 {
                return Err("checkpoint name too long".into());
            }
            let name = std::str::from_utf8(reader.take(len)?)
                .map_err(|_| "invalid checkpoint name encoding")?
                .to_owned();
            if !valid_name(&name) {
                return Err("invalid checkpoint parameter name".into());
            }
            let rank = reader.u32()? as usize;
            if rank > 8 {
                return Err("checkpoint rank exceeds 8".into());
            }
            let mut shape = vec![];
            for _ in 0..rank {
                shape.push(
                    usize::try_from(reader.u64()?).map_err(|_| "checkpoint dimension overflow")?,
                );
            }
            let n = usize::try_from(reader.u64()?).map_err(|_| "checkpoint size overflow")?;
            if tensor::elements(&shape)? != n {
                return Err("checkpoint shape/data size mismatch".into());
            }
            total = total.checked_add(n).ok_or("checkpoint size overflow")?;
            if total > tensor::MAX_ELEMENTS {
                return Err("checkpoint exceeds storage limit".into());
            }
            let data = reader.floats(n)?;
            let t = Tensor::new(shape, data)?;
            if checkpoint.parameters.insert(name.clone(), t).is_some() {
                return Err("duplicate checkpoint parameter name".into());
            }
            if reader.flag()? {
                total = total
                    .checked_add(n.checked_mul(2).ok_or("checkpoint size overflow")?)
                    .ok_or("checkpoint size overflow")?;
                if total > tensor::MAX_ELEMENTS {
                    return Err("checkpoint exceeds storage limit".into());
                }
                let state = Moments {
                    step: reader.u64()?,
                    first: reader.floats(n)?,
                    second: reader.floats(n)?,
                };
                checkpoint.adam.insert(name, state);
            }
        }
        if reader.pos != payload.len() {
            return Err("unexpected trailing checkpoint data".into());
        }
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if file.metadata().map_err(|e| e.to_string())?.len() > MAX_FILE as u64 {
            return Err("checkpoint file exceeds size limit".into());
        }
        let mut bytes = vec![];
        file.take((MAX_FILE + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        Self::decode(&bytes)
    }
    /// Publish an entirely written file without replacing any existing destination.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let bytes = self.encode()?;
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = None;
        for _ in 0..32 {
            let candidate = parent.join(format!(
                ".ml-lang-{}-{}.tmp",
                std::process::id(),
                TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(file) => {
                    temporary = Some((candidate, file));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        let (temp, mut file) = temporary.ok_or("cannot create temporary checkpoint file")?;
        let result = (|| {
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            drop(file);
            fs::hard_link(&temp, path).map_err(|e| {
                format!(
                    "cannot save {} (existing destinations are preserved): {e}",
                    path.display()
                )
            })?;
            Ok(())
        })();
        let _ = fs::remove_file(temp);
        result
    }
}
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    })
}
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or("checkpoint length overflow")?;
        let data = self
            .bytes
            .get(self.pos..end)
            .ok_or("truncated checkpoint")?;
        self.pos = end;
        Ok(data)
    }
    fn flag(&mut self) -> Result<bool, String> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid checkpoint flag".into()),
        }
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn float(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn floats(&mut self, n: usize) -> Result<Vec<f64>, String> {
        let bytes = self.take(n.checked_mul(8).ok_or("checkpoint length overflow")?)?;
        Ok(bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
}
