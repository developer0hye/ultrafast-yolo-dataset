// SPDX-License-Identifier: AGPL-3.0-only
//! Input snapshots. Digests cover the exact owned bytes passed to the parser.
use pyo3::{
    exceptions::{PyOSError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyBytes,
};
use rayon::prelude::*;
use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    fs::{self, File, Metadata},
    io::{self, Read},
    sync::atomic::{AtomicUsize, Ordering},
    time::UNIX_EPOCH,
};

thread_local! {
    // Initialized on the first content read on each worker, then reused. Only
    // the prefix actually filled by File::read is hashed or copied. Returned
    // snapshots own separate storage and never borrow this scratch buffer.
    static READ_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

pyo3::create_exception!(_native, InputChangedError, PyRuntimeError);
pyo3::create_exception!(_native, SnapshotLimitError, PyValueError);

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Changed(String),
    Limit(String),
    Memory(String),
}
impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
impl From<Error> for PyErr {
    fn from(value: Error) -> Self {
        match value {
            Error::Io(e) => PyOSError::new_err(e.to_string()),
            Error::Changed(p) => {
                InputChangedError::new_err(format!("input changed during read: {p}"))
            }
            Error::Limit(p) => {
                SnapshotLimitError::new_err(format!("input exceeds snapshot byte limit: {p}"))
            }
            Error::Memory(message) => pyo3::exceptions::PyMemoryError::new_err(message),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Stamp {
    size: u64,
    modified_ns: i128,
    regular: bool,
    identity: [u64; 4],
}
fn stamp(meta: &Metadata) -> io::Result<Stamp> {
    let time = meta.modified()?;
    let modified_ns = match time.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_nanos() as i128,
        Err(e) => -(e.duration().as_nanos() as i128),
    };
    #[cfg(unix)]
    let identity = {
        use std::os::unix::fs::MetadataExt;
        [
            meta.dev(),
            meta.ino(),
            meta.ctime() as u64,
            meta.ctime_nsec() as u64,
        ]
    };
    #[cfg(not(unix))]
    let identity = [0; 4];
    Ok(Stamp {
        size: meta.len(),
        modified_ns,
        regular: meta.is_file(),
        identity,
    })
}

#[derive(Clone, Debug, Serialize)]
pub struct Fingerprint {
    pub kind: &'static str,
    pub size: u64,
    pub modified_ns: i128,
    #[serde(serialize_with = "serialize_digest")]
    pub sha256: Option<[u8; 32]>,
}

// Keep the existing lowercase hexadecimal JSON boundary while retaining the
// native digest without a per-file String allocation or an encode/decode trip.
fn serialize_digest<S: Serializer>(digest: &Option<[u8; 32]>, serializer: S) -> Result<S::Ok, S::Error> {
    match digest {
        None => serializer.serialize_none(),
        Some(bytes) => {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            let mut encoded = [0_u8; 64];
            for (byte, pair) in bytes.iter().zip(encoded.chunks_exact_mut(2)) {
                pair[0] = HEX[(byte >> 4) as usize];
                pair[1] = HEX[(byte & 15) as usize];
            }
            serializer.serialize_some(std::str::from_utf8(&encoded).expect("hex digits are ASCII"))
        }
    }
}

#[pyclass(frozen)]
pub struct Snapshot {
    pub fingerprint: Fingerprint,
    pub data: Option<Vec<u8>>,
}

#[pymethods]
impl Snapshot {
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.data.as_ref().map(|data| PyBytes::new(py, data))
    }
    #[getter]
    fn kind(&self) -> &str {
        self.fingerprint.kind
    }
    fn fingerprint_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fingerprint).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

pub fn read(path: &str, limit: usize, retain: bool, content: bool) -> Result<Snapshot, Error> {
    read_impl(path, limit, retain, content, || Ok(()))
}

fn read_impl(
    path: &str,
    limit: usize,
    retain: bool,
    content: bool,
    after_open: impl FnOnce() -> io::Result<()>,
) -> Result<Snapshot, Error> {
    let meta = match fs::metadata(path) {
        Ok(value) => value,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Ok(Snapshot {
                fingerprint: Fingerprint {
                    kind: "missing",
                    size: 0,
                    modified_ns: 0,
                    sha256: None,
                },
                data: None,
            })
        }
        Err(e) => return Err(e.into()),
    };
    let before = stamp(&meta)?;
    let kind = if meta.is_file() {
        "file"
    } else if meta.is_dir() {
        "directory"
    } else {
        "other"
    };
    if !meta.is_file() || (!retain && !content) {
        return Ok(Snapshot {
            fingerprint: Fingerprint {
                kind,
                size: before.size,
                modified_ns: before.modified_ns,
                sha256: None,
            },
            data: None,
        });
    }
    if before.size > limit as u64 {
        return Err(Error::Limit(path.into()));
    }
    let mut file = File::open(path)?;
    if stamp(&file.metadata()?)? != before {
        return Err(Error::Changed(path.into()));
    }
    after_open()?;
    let mut data = Vec::new();
    if retain {
        data.try_reserve(before.size as usize)
            .map_err(|e| Error::Memory(e.to_string()))?;
    }
    let mut hasher = Sha256::new();
    let mut total = 0_usize;
    READ_BUFFER.with(|scratch| -> Result<(), Error> {
        let mut buffer = scratch.borrow_mut();
        if buffer.is_empty() {
            buffer
                .try_reserve_exact(64 * 1024)
                .map_err(|e| Error::Memory(e.to_string()))?;
            buffer.resize(64 * 1024, 0);
        }
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| Error::Limit(path.into()))?;
            if total > limit {
                return Err(Error::Limit(path.into()));
            }
            hasher.update(&buffer[..count]);
            if retain {
                data.extend_from_slice(&buffer[..count]);
            }
        }
        Ok(())
    })?;
    // Both the opened handle and the current path must still identify the
    // captured file. A rename/replacement cannot silently mix two snapshots.
    let after = fs::metadata(path).map_err(|_| Error::Changed(path.into()))?;
    if stamp(&file.metadata()?)? != before
        || stamp(&after)? != before
        || total as u64 != before.size
    {
        return Err(Error::Changed(path.into()));
    }
    Ok(Snapshot {
        fingerprint: Fingerprint {
            kind,
            size: total as u64,
            modified_ns: before.modified_ns,
            sha256: Some(hasher.finalize().into()),
        },
        data: retain.then_some(data),
    })
}

#[pyfunction]
pub fn read_snapshot(py: Python<'_>, path: String, max_bytes: usize) -> PyResult<Snapshot> {
    if max_bytes == 0 {
        return Err(PyValueError::new_err("max_bytes must be positive"));
    }
    Ok(py.allow_threads(|| read(&path, max_bytes, true, true))?)
}

#[pyfunction]
pub fn fingerprint_paths(
    py: Python<'_>,
    paths: Vec<String>,
    max_bytes: usize,
    workers: usize,
    content: bool,
) -> PyResult<String> {
    crate::validate(1, workers)?;
    if max_bytes == 0 {
        return Err(PyValueError::new_err("max_bytes must be positive"));
    }
    let pool = py
        .allow_threads(|| rayon::ThreadPoolBuilder::new().num_threads(workers).build())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut results = Vec::with_capacity(paths.len());
    for chunk in paths.chunks(256) {
        py.check_signals()?;
        let values: Result<Vec<_>, Error> = py.allow_threads(|| {
            pool.install(|| {
                chunk
                    .par_iter()
                    .map(|p| read(p, max_bytes, false, content).map(|s| s.fingerprint))
                    .collect()
            })
        });
        results.extend(values?);
    }
    serde_json::to_string(&results).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(frozen)]
pub struct CapturedLabels {
    packed: crate::PackedLabels,
    fingerprints: Vec<Fingerprint>,
    fallback: Vec<Option<Vec<u8>>>,
}
#[pymethods]
impl CapturedLabels {
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        self.packed.to_arrays(py)
    }
    fn fingerprints_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.fingerprints).map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn fallback_bytes<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let value = self
            .fallback
            .get(index)
            .ok_or_else(|| pyo3::exceptions::PyIndexError::new_err(index))?;
        Ok(value.as_ref().map(|v| PyBytes::new(py, v)))
    }
}

#[pyfunction]
pub fn snapshot_labels(
    py: Python<'_>,
    paths: Vec<String>,
    num_classes: usize,
    single_cls: bool,
    workers: usize,
    max_file_bytes: usize,
    max_fallback_bytes: usize,
) -> PyResult<CapturedLabels> {
    crate::validate(num_classes, workers)?;
    if max_file_bytes == 0 {
        return Err(PyValueError::new_err("max_file_bytes must be positive"));
    }
    let pool = py
        .allow_threads(|| rayon::ThreadPoolBuilder::new().num_threads(workers).build())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let mut result = CapturedLabels {
        packed: crate::PackedLabels::new(std::iter::empty()),
        fingerprints: vec![],
        fallback: vec![],
    };
    let fallback_bytes = AtomicUsize::new(0);
    for chunk in paths.chunks(256) {
        py.check_signals()?;
        let values: Result<Vec<_>, Error> = py.allow_threads(|| {
            pool.install(|| {
                chunk
                    .par_iter()
                    .map(|path| {
                        let snapshot = read(path, max_file_bytes, true, true)?;
                        let record = match snapshot.data.as_ref() {
                            Some(data) => crate::parser::parse(data, num_classes, single_cls),
                            None => crate::parser::Record::status(1),
                        };
                        let data = if record.status == 3 {
                            snapshot.data
                        } else {
                            None
                        };
                        if let Some(bytes) = data.as_ref() {
                            // Bound accepted buffers before collecting a full
                            // parallel chunk, not after its allocation peak.
                            if fallback_bytes
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                                    total
                                        .checked_add(bytes.len())
                                        .filter(|n| *n <= max_fallback_bytes)
                                })
                                .is_err()
                            {
                                return Err(Error::Limit(
                                    "fallback snapshots exceed max_fallback_bytes".into(),
                                ));
                            }
                        }
                        Ok((record, snapshot.fingerprint, data))
                    })
                    .collect()
            })
        });
        for (record, fingerprint, fallback) in values? {
            result.packed.extend(std::iter::once(record));
            result.fingerprints.push(fingerprint);
            result.fallback.push(fallback);
        }
    }
    Ok(result)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("InputChangedError", m.py().get_type::<InputChangedError>())?;
    m.add(
        "SnapshotLimitError",
        m.py().get_type::<SnapshotLimitError>(),
    )?;
    m.add_class::<Snapshot>()?;
    m.add_class::<CapturedLabels>()?;
    m.add_function(wrap_pyfunction!(read_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(fingerprint_paths, m)?)?;
    m.add_function(wrap_pyfunction!(snapshot_labels, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    #[test]
    fn failed_read_releases_scratch_before_the_next_snapshot() {
        let directory = std::env::temp_dir().join(format!(
            "uyd-snapshot-recovery-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("growing.bin");
        fs::write(&path, b"a").unwrap();
        let failed = read_impl(path.to_str().unwrap(), 4, false, true, || {
            fs::write(&path, b"longer than the limit")
        });
        assert!(matches!(failed, Err(Error::Limit(_))));
        fs::write(&path, b"z").unwrap();
        let recovered = read(path.to_str().unwrap(), 4, true, true).unwrap();
        assert_eq!(recovered.data.as_deref(), Some(&b"z"[..]));
        let expected: [u8; 32] = Sha256::digest(b"z").into();
        assert_eq!(recovered.fingerprint.sha256, Some(expected));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn replacement_during_open_read_is_detected() {
        let directory = std::env::temp_dir().join(format!(
            "uyd-snapshot-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("label.txt");
        fs::write(&path, b"old bytes").unwrap();
        let before = path.metadata().unwrap().modified().unwrap();
        let result = read_impl(path.to_str().unwrap(), 1024, true, true, || {
            fs::rename(&path, directory.join("old.txt"))?;
            fs::write(&path, b"new bytes")?;
            File::open(&path)?.set_modified(before + Duration::from_secs(2))?;
            Ok(())
        });
        assert!(matches!(result, Err(Error::Changed(_))));
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn binary_digest_preserves_exact_json_representation() {
        let fingerprint = Fingerprint {
            kind: "file",
            size: 3,
            modified_ns: -123,
            sha256: Some(Sha256::digest(b"abc").into()),
        };
        assert_eq!(
            serde_json::to_string(&fingerprint).unwrap(),
            r#"{"kind":"file","size":3,"modified_ns":-123,"sha256":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}"#
        );
        let missing = Fingerprint {
            kind: "missing",
            size: 0,
            modified_ns: 0,
            sha256: None,
        };
        assert_eq!(
            serde_json::to_string(&missing).unwrap(),
            r#"{"kind":"missing","size":0,"modified_ns":0,"sha256":null}"#
        );
    }
}
