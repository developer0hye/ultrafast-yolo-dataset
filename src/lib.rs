// SPDX-License-Identifier: AGPL-3.0-only
mod cache;
mod materialize;
mod parser;
mod provenance;
mod snapshot;
use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use rayon::prelude::*;
use std::{fs::File, io::Read};

#[pyclass(frozen)]
struct PackedLabels {
    offsets: Vec<i64>,
    labels: Vec<f32>,
    points: Vec<f32>,
    segment_offsets: Vec<i64>,
    segment_objects: Vec<i64>,
    statuses: Vec<u8>,
    duplicates: Vec<i64>,
}

impl PackedLabels {
    fn new(records: impl IntoIterator<Item = parser::Record>) -> Self {
        let mut p = Self {
            offsets: vec![0],
            labels: vec![],
            points: vec![],
            segment_offsets: vec![0],
            segment_objects: vec![],
            statuses: vec![],
            duplicates: vec![],
        };
        p.extend(records);
        p
    }
    fn extend(&mut self, records: impl IntoIterator<Item = parser::Record>) {
        let p = self;
        for record in records {
            let start = p.labels.len() / 5;
            for row in record.rows {
                p.labels.extend_from_slice(&row);
            }
            for (i, segment) in record.segments.iter().enumerate() {
                p.segment_objects.push((start + i) as i64);
                for point in segment {
                    p.points.extend_from_slice(point);
                }
                p.segment_offsets.push((p.points.len() / 2) as i64);
            }
            p.offsets.push((p.labels.len() / 5) as i64);
            p.statuses.push(record.status);
            p.duplicates.push(record.duplicates as i64);
        }
    }
}

#[pymethods]
impl PackedLabels {
    fn __len__(&self) -> usize {
        self.statuses.len()
    }
    fn to_arrays<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        // Explicit owned copies; no view can outlive or mutate the native result.
        let d = PyDict::new(py);
        d.set_item("object_offsets", self.offsets.clone().into_pyarray(py))?;
        d.set_item(
            "labels",
            self.labels
                .clone()
                .into_pyarray(py)
                .reshape([self.labels.len() / 5, 5])?,
        )?;
        d.set_item(
            "segment_points",
            self.points
                .clone()
                .into_pyarray(py)
                .reshape([self.points.len() / 2, 2])?,
        )?;
        d.set_item(
            "segment_offsets",
            self.segment_offsets.clone().into_pyarray(py),
        )?;
        d.set_item(
            "segment_object_indices",
            self.segment_objects.clone().into_pyarray(py),
        )?;
        d.set_item("statuses", self.statuses.clone().into_pyarray(py))?;
        d.set_item("duplicates", self.duplicates.clone().into_pyarray(py))?;
        Ok(d)
    }
    #[getter]
    fn reference_required(&self) -> Vec<usize> {
        self.statuses
            .iter()
            .enumerate()
            .filter_map(|(i, s)| (*s == 3).then_some(i))
            .collect()
    }
}

fn validate(classes: usize, workers: usize) -> PyResult<()> {
    if classes == 0 || workers == 0 || workers > 256 {
        return Err(PyValueError::new_err(
            "positive num_classes and workers in 1..256 required",
        ));
    }
    Ok(())
}

#[pyfunction]
fn parse_texts(
    py: Python<'_>,
    texts: Vec<Vec<u8>>,
    num_classes: usize,
    single_cls: bool,
) -> PyResult<PackedLabels> {
    validate(num_classes, 1)?;
    Ok(py.allow_threads(|| {
        PackedLabels::new(
            texts
                .iter()
                .map(|s| parser::parse(s, num_classes, single_cls)),
        )
    }))
}

#[pyfunction]
fn parse_files(
    py: Python<'_>,
    paths: Vec<String>,
    num_classes: usize,
    single_cls: bool,
    workers: usize,
    max_file_bytes: usize,
) -> PyResult<PackedLabels> {
    validate(num_classes, workers)?;
    if max_file_bytes == 0 || max_file_bytes > isize::MAX as usize - 1 {
        return Err(PyValueError::new_err("invalid max_file_bytes"));
    }
    let pool = py
        .allow_threads(|| rayon::ThreadPoolBuilder::new().num_threads(workers).build())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut result = PackedLabels::new(std::iter::empty());
    // No global pool survives fork. At most 256 unpacked records exist before
    // merging into final storage. A slow input cannot accumulate an unbounded queue.
    for chunk in paths.chunks(256) {
        py.check_signals()?;
        let records: Vec<_> = py.allow_threads(|| {
            pool.install(|| {
                chunk
                    .par_iter()
                    .map(|path| {
                        if !std::path::Path::new(path).is_file() {
                            return parser::Record::status(1);
                        }
                        let file = match File::open(path) {
                            Ok(f) => f,
                            Err(_) => return parser::Record::status(3),
                        };
                        let mut data = Vec::new();
                        if file
                            .take(max_file_bytes as u64 + 1)
                            .read_to_end(&mut data)
                            .is_err()
                        {
                            return parser::Record::status(3);
                        }
                        if data.len() > max_file_bytes {
                            return parser::Record::status(4);
                        }
                        parser::parse(&data, num_classes, single_cls)
                    })
                    .collect()
            })
        });
        result.extend(records);
    }
    Ok(result)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    snapshot::register(m)?;
    provenance::register(m)?;
    cache::register(m)?;
    materialize::register(m)?;
    m.add_class::<PackedLabels>()?;
    m.add_function(wrap_pyfunction!(parse_texts, m)?)?;
    m.add_function(wrap_pyfunction!(parse_files, m)?)?;
    Ok(())
}
