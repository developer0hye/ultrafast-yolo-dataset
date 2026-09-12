// SPDX-License-Identifier: AGPL-3.0-only
//! Fixed-width fingerprints: no Python object graph is needed on a cache hit.
use pyo3::{exceptions::PyValueError, prelude::*, types::PyBytes};
use rayon::prelude::*;
use serde::Deserialize;

const WIDTH: usize = 57; // kind:u8, size:u64, modified_ns:i128, SHA256:32 bytes
const MAX_INPUTS: usize = 1_000_000;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    kind: String,
    size: u64,
    modified_ns: i128,
    #[serde(deserialize_with = "required_digest")]
    sha256: Option<String>,
}

fn required_digest<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    Option::<String>::deserialize(deserializer)
}

fn record(kind: &str, size: u64, modified_ns: i128, digest: Option<&str>) -> PyResult<[u8; WIDTH]> {
    let mut out = [0; WIDTH];
    out[0] = match kind {
        "file" => 0,
        "missing" => 1,
        "directory" => 2,
        "other" => 3,
        _ => return Err(PyValueError::new_err("unknown fingerprint kind")),
    };
    out[1..9].copy_from_slice(&size.to_le_bytes());
    out[9..25].copy_from_slice(&modified_ns.to_le_bytes());
    if kind == "file" {
        let hex = digest.ok_or_else(|| PyValueError::new_err("missing file digest"))?;
        if hex.len() != 64 {
            return Err(PyValueError::new_err("invalid digest length"));
        }
        fn nibble(c: u8) -> PyResult<u8> {
            match c {
                b'0'..=b'9' => Ok(c - b'0'),
                b'a'..=b'f' => Ok(c - b'a' + 10),
                _ => Err(PyValueError::new_err("invalid digest character")),
            }
        }
        for (target, pair) in out[25..].iter_mut().zip(hex.as_bytes().as_chunks::<2>().0) {
            *target = nibble(pair[0])? * 16 + nibble(pair[1])?;
        }
    } else if digest.is_some() || (kind == "missing" && (size != 0 || modified_ns != 0)) {
        return Err(PyValueError::new_err("invalid non-file fingerprint"));
    }
    Ok(out)
}

fn check(data: &[u8], count: usize) -> PyResult<()> {
    if count > MAX_INPUTS || data.len() != count * WIDTH {
        return Err(PyValueError::new_err("invalid fingerprint table length"));
    }
    for row in data.as_chunks::<WIDTH>().0 {
        if row[0] > 3
            || (row[0] != 0 && row[25..].iter().any(|&v| v != 0))
            || (row[0] == 1 && row[1..25].iter().any(|&v| v != 0))
        {
            return Err(PyValueError::new_err("invalid fingerprint table record"));
        }
    }
    Ok(())
}

#[pyfunction]
fn pack_fingerprint_table<'py>(
    py: Python<'py>,
    json: &str,
    count: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if count > MAX_INPUTS {
        return Err(PyValueError::new_err("too many fingerprints"));
    }
    let data = py.allow_threads(|| -> PyResult<_> {
        let entries: Vec<Entry> =
            serde_json::from_str(json).map_err(|e| PyValueError::new_err(e.to_string()))?;
        if entries.len() != count {
            return Err(PyValueError::new_err("invalid fingerprint count"));
        }
        let mut data = Vec::new();
        data.try_reserve_exact(count * WIDTH)
            .map_err(|e| pyo3::exceptions::PyMemoryError::new_err(e.to_string()))?;
        for entry in entries {
            data.extend_from_slice(&record(
                &entry.kind,
                entry.size,
                entry.modified_ns,
                entry.sha256.as_deref(),
            )?);
        }
        Ok(data)
    })?;
    Ok(PyBytes::new(py, &data))
}

#[pyfunction]
fn check_fingerprint_table(
    py: Python<'_>,
    table: &Bound<'_, PyBytes>,
    count: usize,
) -> PyResult<()> {
    let data = table.as_bytes();
    py.allow_threads(|| check(data, count))
}

#[pyfunction]
fn verify_fingerprint_table(
    py: Python<'_>,
    table: &Bound<'_, PyBytes>,
    paths: Vec<String>,
    max_bytes: usize,
    workers: usize,
    content: bool,
) -> PyResult<()> {
    crate::validate(1, workers)?;
    if max_bytes == 0 {
        return Err(PyValueError::new_err("max_bytes must be positive"));
    }
    let data = table.as_bytes();
    py.allow_threads(|| check(data, paths.len()))?;
    let pool = py
        .allow_threads(|| rayon::ThreadPoolBuilder::new().num_threads(workers).build())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    for (inputs, expected) in paths.chunks(256).zip(data.chunks(256 * WIDTH)) {
        py.check_signals()?;
        py.allow_threads(|| {
            pool.install(|| {
                inputs
                    .par_iter()
                    .zip(expected.par_chunks_exact(WIDTH))
                    .try_for_each(|(path, stored)| -> PyResult<()> {
                        let value =
                            crate::snapshot::read(path, max_bytes, false, content)?.fingerprint;
                        // Metadata checks deliberately ignore SHA256. The canonical
                        // record still requires a digest for files, so use zero bytes.
                        let digest = if content {
                            value.sha256.as_deref()
                        } else if value.kind == "file" {
                            Some("0000000000000000000000000000000000000000000000000000000000000000")
                        } else {
                            None
                        };
                        let current = record(value.kind, value.size, value.modified_ns, digest)?;
                        let width = if content { WIDTH } else { 25 };
                        if current[..width] != stored[..width] {
                            return Err(crate::snapshot::InputChangedError::new_err(format!(
                                "input differs from captured bytes: {path}"
                            )));
                        }
                        Ok(())
                    })
            })
        })?;
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_fingerprint_table, m)?)?;
    m.add_function(wrap_pyfunction!(check_fingerprint_table, m)?)?;
    m.add_function(wrap_pyfunction!(verify_fingerprint_table, m)?)?;
    Ok(())
}
