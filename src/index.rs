// SPDX-License-Identifier: AGPL-3.0-only
//! Fused image discovery, label-path derivation and per-file metadata.
//!
//! The reference pays Python per-object costs for every `os.stat` result, and
//! its GIL serializes one metadata syscall per file. Here each directory is
//! listed once together with its entries' metadata (`getattrlistbulk` on
//! macOS), directories are processed in parallel without the GIL, paths live
//! in one contiguous arena, and only the final path strings cross into Python.
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyBytes, PyList, PyString},
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io,
    sync::atomic::{AtomicBool, Ordering},
};

pub const KIND_FILE: u8 = 0;
pub const KIND_MISSING: u8 = 1;
pub const KIND_DIR: u8 = 2;
pub const KIND_OTHER: u8 = 3;
const DIGEST_CHUNK: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Meta {
    pub kind: u8,
    pub size: u64,
    pub mtime_ns: i64,
}

impl Meta {
    const MISSING: Meta = Meta {
        kind: KIND_MISSING,
        size: 0,
        mtime_ns: 0,
    };
}

/// One directory entry as reported by the platform listing.
struct RawEntry {
    name: Vec<u8>,
    /// `None` when the listing could not provide metadata (errors, symlinks).
    meta: Option<Meta>,
    is_symlink: bool,
}

fn from_std(meta: &std::fs::Metadata) -> Meta {
    #[cfg(unix)]
    let mtime_ns = {
        use std::os::unix::fs::MetadataExt;
        meta.mtime()
            .saturating_mul(1_000_000_000)
            .saturating_add(meta.mtime_nsec())
    };
    #[cfg(not(unix))]
    let mtime_ns = match meta.modified().map(|t| t.duration_since(std::time::UNIX_EPOCH)) {
        Ok(Ok(d)) => d.as_nanos() as i64,
        Ok(Err(e)) => -(e.duration().as_nanos() as i64),
        Err(_) => 0,
    };
    let kind = if meta.is_file() {
        KIND_FILE
    } else if meta.is_dir() {
        KIND_DIR
    } else {
        KIND_OTHER
    };
    Meta {
        kind,
        size: if kind == KIND_DIR { 0 } else { meta.len() },
        mtime_ns,
    }
}

/// `os.stat` semantics: follow symlinks; any failure is a missing entry.
fn stat_follow(path: &str) -> Meta {
    std::fs::metadata(path)
        .map(|m| from_std(&m))
        .unwrap_or(Meta::MISSING)
}

#[cfg(target_os = "macos")]
mod platform {
    use super::*;
    use std::{ffi::CString, mem::size_of};

    // Not exported by libc: <sys/attr.h>. Returned directly after
    // ATTR_CMN_RETURNED_ATTRS when requested (see getattrlistbulk(2)).
    const ATTR_CMN_ERROR: u32 = 0x2000_0000;
    const VREG: u32 = 1;
    const VDIR: u32 = 2;
    const VLNK: u32 = 5;

    struct Descriptor(libc::c_int);
    impl Drop for Descriptor {
        fn drop(&mut self) {
            unsafe { libc::close(self.0) };
        }
    }

    fn read<T: Copy>(buffer: &[u8], offset: usize) -> io::Result<T> {
        if offset.checked_add(size_of::<T>()).is_none_or(|end| end > buffer.len()) {
            return Err(io::Error::other("truncated getattrlistbulk record"));
        }
        // SAFETY: bounds checked above; the kernel packs fields without Rust alignment.
        Ok(unsafe { std::ptr::read_unaligned(buffer.as_ptr().add(offset) as *const T) })
    }

    pub fn list(dir: &str) -> io::Result<Vec<RawEntry>> {
        let path = CString::new(dir).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let fd = unsafe {
            libc::open(
                path.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        let descriptor = Descriptor(fd);
        let mut request: libc::attrlist = unsafe { std::mem::zeroed() };
        request.bitmapcount = libc::ATTR_BIT_MAP_COUNT;
        request.commonattr = libc::ATTR_CMN_RETURNED_ATTRS
            | libc::ATTR_CMN_NAME
            | ATTR_CMN_ERROR
            | libc::ATTR_CMN_OBJTYPE
            | libc::ATTR_CMN_MODTIME;
        request.fileattr = libc::ATTR_FILE_DATALENGTH;
        let mut buffer = vec![0u8; 256 * 1024];
        let mut entries = Vec::new();
        loop {
            let count = unsafe {
                libc::getattrlistbulk(
                    descriptor.0,
                    &mut request as *mut libc::attrlist as *mut libc::c_void,
                    buffer.as_mut_ptr() as *mut libc::c_void,
                    buffer.len(),
                    0,
                )
            };
            if count < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if count == 0 {
                break;
            }
            let mut entry = 0usize;
            for _ in 0..count {
                let length = read::<u32>(&buffer, entry)? as usize;
                if length < 4 || entry + length > buffer.len() {
                    return Err(io::Error::other("invalid getattrlistbulk record length"));
                }
                let record = &buffer[entry..entry + length];
                let mut field = 4;
                let returned: libc::attribute_set_t = read(record, field)?;
                field += size_of::<libc::attribute_set_t>();
                let mut error = 0u32;
                if returned.commonattr & ATTR_CMN_ERROR != 0 {
                    error = read(record, field)?;
                    field += 4;
                }
                let mut name = None;
                if returned.commonattr & libc::ATTR_CMN_NAME != 0 {
                    let reference: libc::attrreference_t = read(record, field)?;
                    let start = field as isize + reference.attr_dataoffset as isize;
                    let length = reference.attr_length as usize;
                    if start < 0 || length == 0 || start as usize + length > record.len() {
                        return Err(io::Error::other("invalid getattrlistbulk name"));
                    }
                    // attr_length includes the terminating NUL.
                    name = Some(record[start as usize..start as usize + length - 1].to_vec());
                    field += size_of::<libc::attrreference_t>();
                }
                let mut object = None;
                if returned.commonattr & libc::ATTR_CMN_OBJTYPE != 0 {
                    object = Some(read::<u32>(record, field)?);
                    field += 4;
                }
                let mut modified = None;
                if returned.commonattr & libc::ATTR_CMN_MODTIME != 0 {
                    let time: libc::timespec = read(record, field)?;
                    field += size_of::<libc::timespec>();
                    modified = Some(
                        (time.tv_sec as i64)
                            .saturating_mul(1_000_000_000)
                            .saturating_add(time.tv_nsec as i64),
                    );
                }
                let mut length_bytes = None;
                if returned.fileattr & libc::ATTR_FILE_DATALENGTH != 0 {
                    length_bytes = Some(read::<libc::off_t>(record, field)?);
                }
                entry += length;
                let Some(name) = name else { continue };
                let is_symlink = object == Some(VLNK);
                let meta = match (error, object, modified) {
                    (0, Some(VREG), Some(mtime_ns)) => length_bytes.map(|size| Meta {
                        kind: KIND_FILE,
                        size: size.max(0) as u64,
                        mtime_ns,
                    }),
                    (0, Some(VDIR), Some(mtime_ns)) => Some(Meta {
                        kind: KIND_DIR,
                        size: 0,
                        mtime_ns,
                    }),
                    (0, Some(kind), Some(mtime_ns)) if kind != VLNK => Some(Meta {
                        kind: KIND_OTHER,
                        size: length_bytes.unwrap_or(0).max(0) as u64,
                        mtime_ns,
                    }),
                    _ => None,
                };
                entries.push(RawEntry {
                    name,
                    meta,
                    is_symlink,
                });
            }
        }
        Ok(entries)
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use super::*;

    pub fn list(dir: &str) -> io::Result<Vec<RawEntry>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            // glob swallows per-entry errors; so does this listing.
            let Ok(entry) = entry else { continue };
            let Ok(kind) = entry.file_type() else { continue };
            #[cfg(unix)]
            let name = {
                use std::os::unix::ffi::OsStrExt;
                entry.file_name().as_bytes().to_vec()
            };
            #[cfg(not(unix))]
            let name = match entry.file_name().into_string() {
                Ok(s) => s.into_bytes(),
                Err(_) => vec![0xff],
            };
            // Linux stat is cheap; metadata is resolved lazily for matches.
            let meta = kind.is_dir().then_some(Meta {
                kind: KIND_DIR,
                size: 0,
                mtime_ns: 0,
            });
            entries.push(RawEntry {
                name,
                meta,
                is_symlink: kind.is_symlink(),
            });
        }
        Ok(entries)
    }
}

/// Replicates `glob.glob(root/**/*.*, recursive=True)` then the reference's
/// extension filter. Returns `Err(())` for layouts whose exact glob semantics
/// are not replicated (symlinked directories, undecodable names).
fn walk(
    directory: &str,
    formats: &[Vec<u8>],
    abort: &AtomicBool,
) -> Vec<(String, Meta)> {
    if abort.load(Ordering::Relaxed) {
        return Vec::new();
    }
    // glob's _iterdir swallows listing errors (e.g. permissions).
    let Ok(entries) = platform::list(directory) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    let mut subdirectories = Vec::new();
    for entry in entries {
        // glob skips hidden names in both `**` recursion and `*.*` matching.
        if entry.name.first() == Some(&b'.') {
            continue;
        }
        let Ok(name) = std::str::from_utf8(&entry.name) else {
            abort.store(true, Ordering::Relaxed);
            return Vec::new();
        };
        let path = format!("{directory}/{name}");
        let meta = if entry.is_symlink {
            let meta = stat_follow(&path);
            if meta.kind == KIND_DIR {
                // glob would recurse through it (and possibly loop).
                abort.store(true, Ordering::Relaxed);
                return Vec::new();
            }
            Some(meta)
        } else {
            entry.meta
        };
        let is_dir = meta.is_some_and(|m| m.kind == KIND_DIR);
        // fnmatch("*.*") on a non-hidden name: at least one dot. The reference
        // then lowercases the text after the path's last dot.
        let matches = name.rfind('.').is_some_and(|dot| {
            let extension = &name.as_bytes()[dot + 1..];
            formats
                .iter()
                .any(|f| f.len() == extension.len() && f.eq_ignore_ascii_case(extension))
        });
        if matches {
            let meta = match meta {
                Some(m) if m.kind != KIND_DIR || m.mtime_ns != 0 => m,
                _ => stat_follow(&path),
            };
            found.push((path.clone(), meta));
        }
        if is_dir {
            subdirectories.push(path);
        }
    }
    let nested: Vec<Vec<(String, Meta)>> = subdirectories
        .par_iter()
        .map(|child| walk(child, formats, abort))
        .collect();
    for part in nested {
        found.extend(part);
    }
    found
}

/// `img2label_paths`: replace the last `/images/` and the final extension.
fn label_path(image: &str, images: &str, labels: &str, suffix: &str) -> String {
    let replaced = match image.rfind(images) {
        Some(i) => {
            let mut s = String::with_capacity(image.len() + labels.len());
            s.push_str(&image[..i]);
            s.push_str(labels);
            s.push_str(&image[i + images.len()..]);
            s
        }
        None => image.to_owned(),
    };
    let stem = match replaced.rfind('.') {
        Some(dot) => &replaced[..dot],
        None => &replaced[..],
    };
    let mut out = String::with_capacity(stem.len() + suffix.len());
    out.push_str(stem);
    out.push_str(suffix);
    out
}

fn split_parent(path: &str) -> (&str, &str) {
    match path.rfind('/') {
        Some(0) => ("/", &path[1..]),
        Some(i) => (&path[..i], &path[i + 1..]),
        None => (".", path),
    }
}

/// Metadata for arbitrary paths with `os.stat` semantics.
fn stat_many(paths: &[String]) -> Vec<Meta> {
    #[cfg(target_os = "macos")]
    {
        // Group by parent and list each directory once in bulk. Exact-name
        // misses still use stat: APFS lookups may be case/normalization
        // insensitive, and listings report symlinks themselves.
        let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, path) in paths.iter().enumerate() {
            groups.entry(split_parent(path).0).or_default().push(index);
        }
        let groups: Vec<(&str, Vec<usize>)> = groups.into_iter().collect();
        let resolved: Vec<Vec<(usize, Meta)>> = groups
            .par_iter()
            .map(|(parent, indices)| {
                let listing = platform::list(parent).unwrap_or_default();
                let by_name: HashMap<&[u8], &RawEntry> =
                    listing.iter().map(|e| (e.name.as_slice(), e)).collect();
                indices
                    .iter()
                    .map(|&index| {
                        let path = &paths[index];
                        let name = split_parent(path).1.as_bytes();
                        let meta = match by_name.get(name) {
                            Some(entry) if !entry.is_symlink => {
                                entry.meta.unwrap_or_else(|| stat_follow(path))
                            }
                            _ => stat_follow(path),
                        };
                        (index, meta)
                    })
                    .collect()
            })
            .collect();
        let mut out = vec![Meta::MISSING; paths.len()];
        for part in resolved {
            for (index, meta) in part {
                out[index] = meta;
            }
        }
        out
    }
    #[cfg(not(target_os = "macos"))]
    {
        paths.par_iter().map(|p| stat_follow(p)).collect()
    }
}

fn digest(images: &[String], image_meta: &[Meta], labels: &[String], label_meta: &[Meta]) -> [u8; 32] {
    let chunks: Vec<[u8; 32]> = (0..images.len().div_ceil(DIGEST_CHUNK))
        .into_par_iter()
        .map(|chunk| {
            let mut h = Sha256::new();
            let end = ((chunk + 1) * DIGEST_CHUNK).min(images.len());
            for i in chunk * DIGEST_CHUNK..end {
                for (path, meta) in [(&images[i], image_meta[i]), (&labels[i], label_meta[i])] {
                    h.update((path.len() as u64).to_le_bytes());
                    h.update(path.as_bytes());
                    h.update([meta.kind]);
                    h.update(meta.size.to_le_bytes());
                    h.update(meta.mtime_ns.to_le_bytes());
                }
            }
            h.finalize().into()
        })
        .collect();
    let mut h = Sha256::new();
    h.update(b"ultrafast-yolo-dataset-index-v1\0");
    h.update((images.len() as u64).to_le_bytes());
    for c in &chunks {
        h.update(c);
    }
    h.finalize().into()
}

/// Ordered image/label paths with per-file metadata. Frozen and GIL-free.
#[pyclass(frozen)]
pub struct DatasetIndex {
    images: Vec<String>,
    image_meta: Vec<Meta>,
    labels: Vec<String>,
    label_meta: Vec<Meta>,
}

impl DatasetIndex {
    fn build(images: Vec<String>, image_meta: Vec<Meta>, sa: &str, sb: &str, suffix: &str) -> Self {
        let labels: Vec<String> = images
            .par_iter()
            .map(|p| label_path(p, sa, sb, suffix))
            .collect();
        let label_meta = stat_many(&labels);
        Self {
            images,
            image_meta,
            labels,
            label_meta,
        }
    }
}

#[pymethods]
impl DatasetIndex {
    fn __len__(&self) -> usize {
        self.images.len()
    }
    fn image_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.images.iter().map(|p| PyString::new(py, p)))
    }
    fn label_paths<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.labels.iter().map(|p| PyString::new(py, p)))
    }
    /// Ordered digest of paths, kinds, sizes and modification times.
    fn digest<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let value = py.allow_threads(|| digest(&self.images, &self.image_meta, &self.labels, &self.label_meta));
        PyBytes::new(py, &value)
    }
    /// The first `count` pairs, as the reference applies `fraction` after sorting.
    fn truncated(&self, count: usize) -> Self {
        let n = count.min(self.images.len());
        Self {
            images: self.images[..n].to_vec(),
            image_meta: self.image_meta[..n].to_vec(),
            labels: self.labels[..n].to_vec(),
            label_meta: self.label_meta[..n].to_vec(),
        }
    }
    /// (kind, size, mtime_ns) per image and label; diagnostics and tests only.
    fn metadata(&self) -> (Vec<(u8, u64, i64)>, Vec<(u8, u64, i64)>) {
        let unpack = |m: &[Meta]| m.iter().map(|m| (m.kind, m.size, m.mtime_ns)).collect();
        (unpack(&self.image_meta), unpack(&self.label_meta))
    }
}

fn pool(workers: usize) -> PyResult<rayon::ThreadPool> {
    if workers == 0 || workers > 256 {
        return Err(PyValueError::new_err("workers must be in 1..256"));
    }
    // A per-call pool: no global Rayon state is inherited across fork.
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Discover images under one directory exactly as the pinned reference's glob
/// and extension filter would, then index their labels. `None` means the
/// caller must use the reference discovery (unsupported layout).
#[pyfunction]
#[pyo3(signature = (root, formats, workers, images_dir="/images/", labels_dir="/labels/", suffix=".txt"))]
fn discover_dataset(
    py: Python<'_>,
    root: String,
    formats: Vec<String>,
    workers: usize,
    images_dir: &str,
    labels_dir: &str,
    suffix: &str,
) -> PyResult<Option<DatasetIndex>> {
    if root.is_empty() || root.ends_with('/') || root.contains(['*', '?', '[']) {
        return Ok(None);
    }
    let formats: Vec<Vec<u8>> = formats.into_iter().map(|f| f.into_bytes()).collect();
    let pool = pool(workers)?;
    py.allow_threads(|| {
        pool.install(|| {
            let abort = AtomicBool::new(false);
            let mut found = walk(&root, &formats, &abort);
            if abort.load(Ordering::Relaxed) {
                return Ok(None);
            }
            // Python compares str by code point; UTF-8 byte order is identical.
            found.par_sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            let (images, meta): (Vec<String>, Vec<Meta>) = found.into_iter().unzip();
            Ok(Some(DatasetIndex::build(images, meta, images_dir, labels_dir, suffix)))
        })
    })
}

/// Index an explicit ordered image list (file lists, multiple roots).
#[pyfunction]
#[pyo3(signature = (images, workers, images_dir="/images/", labels_dir="/labels/", suffix=".txt"))]
fn index_paths(
    py: Python<'_>,
    images: Vec<String>,
    workers: usize,
    images_dir: &str,
    labels_dir: &str,
    suffix: &str,
) -> PyResult<DatasetIndex> {
    let pool = pool(workers)?;
    Ok(py.allow_threads(|| {
        pool.install(|| {
            let meta = stat_many(&images);
            DatasetIndex::build(images, meta, images_dir, labels_dir, suffix)
        })
    }))
}

fn chunked(bytes: &[u8]) -> [u8; 32] {
    const CHUNK: usize = 1 << 20;
    let parts: Vec<[u8; 32]> = bytes
        .par_chunks(CHUNK)
        .map(|c| Sha256::digest(c).into())
        .collect();
    let mut h = Sha256::new();
    h.update(b"ultrafast-yolo-dataset-chunked-sha256-v1\0");
    h.update((bytes.len() as u64).to_le_bytes());
    for part in &parts {
        h.update(part);
    }
    h.finalize().into()
}

/// SHA-256 over 1 MiB chunk digests, computed in parallel without the GIL.
/// Cache sections are integrity-checked with this instead of one serial hash.
#[pyfunction]
fn chunked_sha256<'py>(
    py: Python<'py>,
    data: pyo3::buffer::PyBuffer<u8>,
    workers: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    if !data.is_c_contiguous() {
        return Err(PyValueError::new_err("a contiguous buffer is required"));
    }
    let length = data.len_bytes();
    let address = data.buf_ptr() as usize;
    // SAFETY: `data` holds the exporter's buffer for this whole call. Callers
    // pass read-only mappings or freshly built arrays that nothing else mutates.
    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, length) };
    let value = if length <= 4 << 20 {
        chunked(bytes)
    } else {
        let pool = pool(workers)?;
        py.allow_threads(|| pool.install(|| chunked(bytes)))
    };
    Ok(PyBytes::new(py, &value))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatasetIndex>()?;
    m.add_function(wrap_pyfunction!(discover_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(index_paths, m)?)?;
    m.add_function(wrap_pyfunction!(chunked_sha256, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_paths_match_reference_string_semantics() {
        let f = |p| label_path(p, "/images/", "/labels/", ".txt");
        assert_eq!(f("/d/images/a/b.jpg"), "/d/labels/a/b.txt");
        assert_eq!(f("/images/x/images/c.d.png"), "/images/x/labels/c.d.txt");
        assert_eq!(f("/d/pics/a.jpg"), "/d/pics/a.txt");
        assert_eq!(f("/d/images/noext"), "/d/labels/noext.txt");
    }

    #[test]
    fn bulk_listing_matches_std_metadata() {
        let dir = std::env::temp_dir().join(format!("uyd-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub.jpg")).unwrap();
        std::fs::write(dir.join("a.JPG"), b"12345").unwrap();
        std::fs::write(dir.join("b.txt"), b"").unwrap();
        std::fs::write(dir.join(".hidden.jpg"), b"x").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.join("a.JPG"), dir.join("link.png")).unwrap();
        let listing = platform::list(dir.to_str().unwrap()).unwrap();
        for entry in &listing {
            let name = String::from_utf8(entry.name.clone()).unwrap();
            let path = dir.join(&name);
            if entry.is_symlink {
                assert_eq!(name, "link.png");
                continue;
            }
            let expected = from_std(&std::fs::metadata(&path).unwrap());
            if let Some(meta) = entry.meta {
                assert_eq!(meta.kind, expected.kind, "{name}");
                if meta.kind != KIND_DIR {
                    assert_eq!(meta, expected, "{name}");
                }
            }
        }
        assert_eq!(listing.len(), 5);
        let abort = AtomicBool::new(false);
        let formats: Vec<Vec<u8>> = ["jpg", "png"].iter().map(|s| s.as_bytes().to_vec()).collect();
        let mut found = walk(dir.to_str().unwrap(), &formats, &abort);
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let names: Vec<&str> = found.iter().map(|(p, _)| p.rsplit('/').next().unwrap()).collect();
        assert!(!abort.load(Ordering::Relaxed));
        assert_eq!(names, ["a.JPG", "link.png", "sub.jpg"]);
        assert_eq!(found[1].1, found[0].1, "symlinks report their target");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
