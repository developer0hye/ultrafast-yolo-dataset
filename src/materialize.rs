// SPDX-License-Identifier: AGPL-3.0-only
//! Construct mutable framework records without per-record Python slicing/dispatch.
use numpy::{
    Element, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray, PyReadonlyArray1,
    PyReadonlyArray2, PyUntypedArrayMethods, PY_ARRAY_API,
};
use pyo3::{
    exceptions::{PyOverflowError, PyValueError},
    prelude::*,
    types::{PyDict, PyList, PyString, PyTuple},
};

fn require(value: bool, message: &str) -> PyResult<()> {
    if value {
        Ok(())
    } else {
        Err(PyValueError::new_err(message.to_owned()))
    }
}

fn matrix(py: Python<'_>, rows: usize, columns: usize) -> PyResult<Bound<'_, PyArray2<f32>>> {
    let mut dims = [
        isize::try_from(rows).map_err(|_| PyOverflowError::new_err("array row count overflow"))?,
        isize::try_from(columns)
            .map_err(|_| PyOverflowError::new_err("array column count overflow"))?,
    ];
    // NumPy owns this initialized C-order allocation directly: no per-array
    // Rust Vec owner/capsule is needed. PyArray_Zeros steals the dtype reference,
    // copies the two dimensions, and returns a new reference (or an exception).
    // Unlike the infallible numpy-crate convenience constructor, propagate an
    // allocation failure as the original Python error instead of panicking.
    let value = unsafe {
        let pointer = PY_ARRAY_API.PyArray_Zeros(
            py,
            2,
            dims.as_mut_ptr(),
            f32::get_dtype(py).into_dtype_ptr(),
            0,
        );
        Bound::<PyAny>::from_owned_ptr_or_err(py, pointer)?
    };
    Ok(value.downcast_into::<PyArray2<f32>>()?)
}

fn layout<T: Element, D: numpy::ndarray::Dimension>(
    array: &PyReadonlyArray<'_, T, D>,
) -> PyResult<()> {
    // numpy 0.25's as_slice accepts both C/F contiguous storage and does not
    // check data alignment. Check both before forming typed Rust slices.
    require(
        array.is_c_contiguous()
            && (array.data() as usize).is_multiple_of(std::mem::align_of::<T>()),
        "packed arrays must be aligned and C-contiguous",
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn materialize_labels<'py>(
    py: Python<'py>,
    image_paths: &Bound<'py, PyTuple>,
    sources: PyReadonlyArray1<'py, i64>,
    shapes: PyReadonlyArray2<'py, i64>,
    offsets: PyReadonlyArray1<'py, i64>,
    labels: PyReadonlyArray2<'py, f32>,
    points: PyReadonlyArray2<'py, f32>,
    segment_offsets: PyReadonlyArray1<'py, i64>,
    objects: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyList>> {
    let source_count = sources.len();
    let rows = labels.shape()[0];
    let point_count = points.shape()[0];
    require(
        shapes.shape() == [source_count, 2] && labels.shape()[1] == 5 && points.shape()[1] == 2,
        "invalid materialization matrix shapes",
    )?;
    layout(&sources)?;
    layout(&shapes)?;
    layout(&offsets)?;
    layout(&labels)?;
    layout(&points)?;
    layout(&segment_offsets)?;
    layout(&objects)?;
    let sources = sources.as_slice()?;
    let shapes = shapes.as_slice()?;
    let offsets = offsets.as_slice()?;
    let labels = labels.as_slice()?;
    let points = points.as_slice()?;
    let segment_offsets = segment_offsets.as_slice()?;
    let objects = objects.as_slice()?;
    require(
        sources
            .iter()
            .all(|&i| i >= 0 && (i as usize) < image_paths.len())
            && sources.windows(2).all(|w| w[0] < w[1]),
        "invalid source indices",
    )?;
    for (values, length, end) in [
        (offsets, source_count + 1, rows),
        (segment_offsets, objects.len() + 1, point_count),
    ] {
        require(
            values.len() == length
                && values.first() == Some(&0)
                && values.last() == Some(&(end as i64))
                && values.windows(2).all(|w| w[0] <= w[1]),
            "invalid materialization offsets",
        )?;
    }
    require(
        objects.iter().all(|&i| i >= 0 && (i as usize) < rows)
            && objects.windows(2).all(|w| w[0] < w[1]),
        "invalid segment object indices",
    )?;

    // The GIL stays held while reading the NumPy buffers and creating Python
    // objects. Every exported numeric array gets its own owned allocation;
    // neither another record nor another export can mutate it through a base.
    let output = PyList::empty(py);
    let mut segment_index = 0;
    for index in 0..source_count {
        if index % 256 == 0 {
            py.check_signals()?;
        }
        let begin = offsets[index] as usize;
        let end = offsets[index + 1] as usize;
        let count = end - begin;
        let classes = matrix(py, count, 1)?;
        let boxes = matrix(py, count, 4)?;
        {
            let mut class_view = classes.readwrite();
            let mut box_view = boxes.readwrite();
            let class_data = class_view.as_slice_mut()?;
            let box_data = box_view.as_slice_mut()?;
            for (index, row) in labels[begin * 5..end * 5]
                .as_chunks::<5>()
                .0
                .iter()
                .enumerate()
            {
                class_data[index] = row[0];
                box_data[index * 4..index * 4 + 4].copy_from_slice(&row[1..]);
            }
        }
        let segments = PyList::empty(py);
        let first_segment = segment_index;
        while segment_index < objects.len() && (objects[segment_index] as usize) < end {
            let a = segment_offsets[segment_index] as usize;
            let b = segment_offsets[segment_index + 1] as usize;
            let polygon = matrix(py, b - a, 2)?;
            polygon
                .readwrite()
                .as_slice_mut()?
                .copy_from_slice(&points[a * 2..b * 2]);
            segments.append(polygon)?;
            segment_index += 1;
        }
        require(
            segment_index == first_segment || segment_index - first_segment == count,
            "partial image segment alignment",
        )?;
        let row = PyDict::new(py);
        let path = image_paths.get_item(sources[index] as usize)?;
        require(
            path.is_instance_of::<PyString>(),
            "image paths must be strings",
        )?;
        row.set_item(pyo3::intern!(py, "im_file"), path)?;
        row.set_item(
            pyo3::intern!(py, "shape"),
            (shapes[index * 2], shapes[index * 2 + 1]),
        )?;
        row.set_item(pyo3::intern!(py, "cls"), classes)?;
        row.set_item(pyo3::intern!(py, "bboxes"), boxes)?;
        row.set_item(pyo3::intern!(py, "segments"), segments)?;
        row.set_item(pyo3::intern!(py, "keypoints"), py.None())?;
        row.set_item(pyo3::intern!(py, "normalized"), true)?;
        row.set_item(pyo3::intern!(py, "bbox_format"), pyo3::intern!(py, "xywh"))?;
        output.append(row)?;
    }
    Ok(output)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(materialize_labels, m)?)?;
    Ok(())
}
