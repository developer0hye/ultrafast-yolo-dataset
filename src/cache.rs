// SPDX-License-Identifier: AGPL-3.0-only
//! Bounded checksummed container and process-owned advisory locks.
use pyo3::{
    exceptions::{PyMemoryError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::{self, Read, Write},
    sync::Mutex,
};

pyo3::create_exception!(_native, CacheFormatError, PyValueError);
const MAGIC: &[u8; 8] = b"UYDCACHE";
const VERSION: u32 = 1;
const MAX_SECTIONS: usize = 16;
const MAX_METADATA: u64 = 256 * 1024 * 1024;

fn malformed(message: impl Into<String>) -> PyErr {
    CacheFormatError::new_err(message.into())
}
fn exact(file: &mut File, target: &mut [u8]) -> PyResult<()> {
    file.read_exact(target).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            malformed("truncated cache")
        } else {
            e.into()
        }
    })
}

#[pyfunction]
pub fn read_cache_sections(
    py: Python<'_>,
    path: String,
    max_bytes: usize,
) -> PyResult<Vec<Py<PyBytes>>> {
    if max_bytes == 0 {
        return Err(PyValueError::new_err("max_bytes must be positive"));
    }
    let (mut file, descriptors, initial) = py.allow_threads(|| -> PyResult<_> {
        if !std::fs::metadata(&path)?.is_file() {
            return Err(malformed("cache must be a regular file"));
        }
        let mut file = File::open(path)?;
        let initial = file.metadata()?;
        if !initial.is_file() || initial.len() > max_bytes as u64 {
            return Err(malformed("cache exceeds size/type limits"));
        }
        let mut header = [0_u8; 16];
        exact(&mut file, &mut header)?;
        if &header[..8] != MAGIC || u32::from_le_bytes(header[8..12].try_into().unwrap()) != VERSION
        {
            return Err(malformed("unsupported cache magic/version"));
        }
        let count = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        if count == 0 || count > MAX_SECTIONS {
            return Err(malformed("invalid section count"));
        }
        let mut total = 16_u64 + count as u64 * 40;
        let mut descriptors = Vec::with_capacity(count);
        for index in 0..count {
            let mut value = [0_u8; 40];
            exact(&mut file, &mut value)?;
            let length = u64::from_le_bytes(value[..8].try_into().unwrap());
            if index == 0 && length > MAX_METADATA {
                return Err(malformed("metadata section exceeds limit"));
            }
            total = total
                .checked_add(length)
                .ok_or_else(|| malformed("section length overflow"))?;
            if total > max_bytes as u64 || total > initial.len() {
                return Err(malformed("section exceeds file bounds"));
            }
            descriptors.push((length as usize, <[u8; 32]>::try_from(&value[8..]).unwrap()));
        }
        if total != initial.len() {
            return Err(malformed("trailing bytes or invalid section sizes"));
        }
        Ok((file, descriptors, initial))
    })?;
    let mut result = Vec::with_capacity(descriptors.len());
    for (length, expected) in descriptors {
        py.check_signals()?;
        let mut data = py.allow_threads(|| -> PyResult<_> {
            let mut data = Vec::new();
            data.try_reserve_exact(length)
                .map_err(|e| PyMemoryError::new_err(e.to_string()))?;
            data.resize(length, 0);
            Ok(data)
        })?;
        let mut hasher = Sha256::new();
        for chunk in data.chunks_mut(4 * 1024 * 1024) {
            py.check_signals()?;
            py.allow_threads(|| -> PyResult<()> {
                exact(&mut file, chunk)?;
                hasher.update(chunk);
                Ok(())
            })?;
        }
        if hasher.finalize()[..] != expected {
            return Err(malformed("section checksum mismatch"));
        }
        result.push(PyBytes::new(py, &data).unbind());
    }
    let final_meta = file.metadata()?;
    if initial.len() != final_meta.len() || initial.modified()? != final_meta.modified()? {
        return Err(malformed("cache modified while reading"));
    }
    Ok(result)
}

#[pyfunction]
pub fn write_cache_sections(
    py: Python<'_>,
    path: String,
    sections: Vec<Py<PyBytes>>,
    max_bytes: usize,
) -> PyResult<()> {
    let buffers: Vec<&[u8]> = sections.iter().map(|s| s.bind(py).as_bytes()).collect();
    if buffers.is_empty() || buffers.len() > MAX_SECTIONS || buffers[0].len() as u64 > MAX_METADATA
    {
        return Err(malformed("section count/metadata limit"));
    }
    let mut total = 16_usize + 40 * buffers.len();
    for data in &buffers {
        total = total
            .checked_add(data.len())
            .ok_or_else(|| malformed("section length overflow"))?;
    }
    if total > max_bytes {
        return Err(malformed("cache exceeds byte limit"));
    }
    // PyBytes are immutable and their owning references remain alive outside
    // allow_threads for the entire write. No extra whole-cache copy is made.
    let mut file = py.allow_threads(|| -> PyResult<_> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(MAGIC)?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(buffers.len() as u32).to_le_bytes())?;
        Ok(file)
    })?;
    for data in &buffers {
        py.check_signals()?;
        py.allow_threads(|| -> PyResult<()> {
            file.write_all(&(data.len() as u64).to_le_bytes())?;
            file.write_all(&Sha256::digest(data))?;
            Ok(())
        })?;
    }
    for data in buffers {
        for chunk in data.chunks(4 * 1024 * 1024) {
            py.check_signals()?;
            py.allow_threads(|| file.write_all(chunk))?;
        }
    }
    py.allow_threads(|| -> PyResult<()> {
        file.flush()?;
        file.sync_all()?;
        Ok(())
    })
}

struct LockState {
    file: Option<File>,
    held: bool,
}
#[pyclass]
pub struct CacheLock {
    state: Mutex<LockState>,
    pid: u32,
}
#[pymethods]
impl CacheLock {
    #[new]
    fn new(py: Python<'_>, path: String) -> PyResult<Self> {
        let file = py.allow_threads(|| {
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)
        })?;
        Ok(Self {
            state: Mutex::new(LockState {
                file: Some(file),
                held: false,
            }),
            pid: std::process::id(),
        })
    }
    fn try_acquire(&self) -> PyResult<bool> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "cache locks cannot be used after fork",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("poisoned cache lock"))?;
        if state.held {
            return Err(PyRuntimeError::new_err("cache lock is not reentrant"));
        }
        let file = state
            .file
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("cache lock already released"))?;
        match fs2::FileExt::try_lock_exclusive(file) {
            Ok(()) => {
                state.held = true;
                Ok(true)
            }
            Err(e) if e.raw_os_error() == fs2::lock_contended_error().raw_os_error() => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
    fn release(&self) -> PyResult<()> {
        if self.pid != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "cache locks cannot be used after fork",
            ));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| PyRuntimeError::new_err("poisoned cache lock"))?;
        if let Some(file) = state.file.take() {
            if state.held {
                fs2::FileExt::unlock(&file)?;
            }
        }
        state.held = false;
        Ok(())
    }
}
impl Drop for CacheLock {
    fn drop(&mut self) {
        if let Ok(state) = self.state.get_mut() {
            if self.pid == std::process::id() && state.held {
                if let Some(file) = state.file.as_ref() {
                    let _ = fs2::FileExt::unlock(file);
                }
            }
        }
    }
}

#[pyfunction]
fn native_cache_profile() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_str!("parser.rs"),
        include_str!("snapshot.rs"),
        include_str!("provenance.rs"),
        include_str!("cache.rs"),
        include_str!("lib.rs"),
        include_str!("../Cargo.toml"),
        include_str!("../Cargo.lock"),
    ] {
        hash.update(source.as_bytes());
    }
    format!("{:x}", hash.finalize())
}
#[pyfunction]
fn sync_directory(py: Python<'_>, path: String) -> PyResult<()> {
    #[cfg(unix)]
    py.allow_threads(|| File::open(path)?.sync_all())?;
    #[cfg(not(unix))]
    let _ = (py, path);
    Ok(())
}
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CacheFormatError", m.py().get_type::<CacheFormatError>())?;
    m.add_class::<CacheLock>()?;
    m.add_function(wrap_pyfunction!(read_cache_sections, m)?)?;
    m.add_function(wrap_pyfunction!(write_cache_sections, m)?)?;
    m.add_function(wrap_pyfunction!(native_cache_profile, m)?)?;
    m.add_function(wrap_pyfunction!(sync_directory, m)?)?;
    Ok(())
}
