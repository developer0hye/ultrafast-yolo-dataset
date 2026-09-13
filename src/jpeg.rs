// SPDX-License-Identifier: AGPL-3.0-only
//! Native JPEG header probe for the reference `check_image` on the happy path.
//!
//! The reference opens every image with Pillow, whose JPEG marker loop is pure
//! Python and therefore serialized by the GIL; for large datasets that loop is
//! the whole cold-scan cost. This probe replicates Pillow 12.1.1's
//! `JpegImageFile._open` marker walk, `exif_size` and the reference's EOI check
//! without the GIL, and accepts a file only when every step is certain to give
//! Pillow's result. Anything unusual — junk bytes, unknown markers, MPO data,
//! Photoshop blocks, multiple frames or EXIF segments, EXIF that Pillow could
//! read only partially, XMP-only orientation, truncation, a missing EOI, tiny
//! or decompression-bomb sizes — is reported for the unchanged Pillow path.
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::fs::File;

pub const ACCEPTED: u8 = 0;
pub const PILLOW: u8 = 1;
const FIRST_READ: usize = 16 * 1024;

enum Walk {
    Done {
        width: u16,
        height: u16,
        orientation: Option<u16>,
    },
    NeedMore,
    Pillow,
}

/// IFD0 Orientation from Pillow's `info["exif"]`. `Err` means Pillow could
/// read the payload only partially, or not as a single SHORT, so its outcome
/// is left to Pillow.
fn exif_orientation(mut data: &[u8]) -> Result<Option<u16>, ()> {
    while data.starts_with(b"Exif\0\0") {
        data = &data[6..];
    }
    if data.is_empty() {
        return Ok(None);
    }
    let little = match data.get(..4) {
        Some(b"II\x2a\x00") => true,
        Some(b"MM\x00\x2a") => false,
        _ => return Err(()),
    };
    let u16_at = |at: usize| {
        data.get(at..at + 2).map(|s| {
            if little {
                u16::from_le_bytes([s[0], s[1]])
            } else {
                u16::from_be_bytes([s[0], s[1]])
            }
        })
    };
    let u32_at = |at: usize| {
        data.get(at..at + 4).map(|s| {
            let b = [s[0], s[1], s[2], s[3]];
            if little {
                u32::from_le_bytes(b)
            } else {
                u32::from_be_bytes(b)
            }
        })
    };
    let ifd = u32_at(4).ok_or(())? as usize;
    let count = u16_at(ifd).ok_or(())? as usize;
    let first = ifd + 2;
    // Every entry and the next-IFD pointer must be present; otherwise Pillow
    // stops early with a warning and the loaded tag set depends on position.
    if first + 12 * count + 4 > data.len() {
        return Err(());
    }
    let mut orientation = None;
    for entry in 0..count {
        let at = first + 12 * entry;
        let (tag, kind, n) = (
            u16_at(at).ok_or(())?,
            u16_at(at + 2).ok_or(())?,
            u32_at(at + 4).ok_or(())? as u64,
        );
        let unit: u64 = match kind {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 | 16 | 17 | 18 => 8,
            _ => return Err(()),
        };
        let size = n.checked_mul(unit).ok_or(())?;
        if size > 4 {
            let offset = u32_at(at + 8).ok_or(())? as u64;
            if offset
                .checked_add(size)
                .is_none_or(|end| end > data.len() as u64)
            {
                return Err(());
            }
        }
        if tag == 274 {
            if kind != 3 || n != 1 || orientation.is_some() {
                return Err(());
            }
            orientation = Some(u16_at(at + 8).ok_or(())?);
        }
    }
    Ok(orientation)
}

/// Pillow's APP13 "8BIM" resource loop. Its struct errors end the loop quietly;
/// only indexing the name length past the segment escapes (as IndexError),
/// which makes Image.open fail. Returns false exactly in that case.
fn photoshop_parses(s: &[u8]) -> bool {
    let mut offset = 14usize;
    while s.get(offset..offset + 4) == Some(b"8BIM") {
        offset += 4;
        if offset + 2 > s.len() {
            return true; // i16 struct.error: break
        }
        offset += 2;
        let Some(&name_len) = s.get(offset) else {
            return false;
        };
        offset += 1 + usize::from(name_len);
        offset += offset & 1;
        if offset + 4 > s.len() {
            return true; // i32 struct.error: break
        }
        let size =
            u32::from_be_bytes([s[offset], s[offset + 1], s[offset + 2], s[offset + 3]]) as usize;
        offset += 4;
        // ResolutionInfo decoding errors are struct.error inside the try: break.
        offset = offset.saturating_add(size);
        offset += offset & 1;
    }
    true
}

/// Replica of JpegImageFile._open's marker loop over the available prefix.
fn walk(buf: &[u8], complete: bool) -> Walk {
    macro_rules! need {
        ($end:expr) => {
            if $end > buf.len() {
                return if complete {
                    Walk::Pillow
                } else {
                    Walk::NeedMore
                };
            }
        };
    }
    need!(3);
    if &buf[..3] != b"\xff\xd8\xff" {
        return Walk::Pillow; // another plugin (or none) decides
    }
    let mut pos = 3;
    let mut current = 0xFFu8; // Pillow sets s = b"\xff" after the magic
    let (mut size, mut sof_count) = ((0u16, 0u16), 0);
    let mut exif: Option<&[u8]> = None;
    let mut xmp = false;
    loop {
        if current != 0xFF {
            return Walk::Pillow; // junk before a marker
        }
        need!(pos + 1);
        let marker = 0xFF00 | u16::from(buf[pos]);
        pos += 1;
        match marker {
            0xFFFF => continue, // fill byte: the next 0xFF starts the marker
            0xFF00 => {
                need!(pos + 1);
                current = buf[pos];
                pos += 1;
                continue;
            }
            0xFFC8 | 0xFFD0..=0xFFD9 | 0xFFF0..=0xFFFD => {}
            0xFFC0..=0xFFFE => {
                need!(pos + 2);
                let length = usize::from(u16::from_be_bytes([buf[pos], buf[pos + 1]]));
                if length < 2 {
                    return Walk::Pillow;
                }
                let segment = pos + 2..pos + length;
                need!(segment.end);
                let s = &buf[segment.clone()];
                pos = segment.end;
                match marker {
                    0xFFC0..=0xFFC3
                    | 0xFFC5..=0xFFC7
                    | 0xFFC9..=0xFFCB
                    | 0xFFCD..=0xFFCF
                    | 0xFFDE => {
                        sof_count += 1;
                        if sof_count > 1 || s.len() < 6 || (s.len() - 6) % 3 != 0 || s[0] != 8 {
                            return Walk::Pillow;
                        }
                        if !matches!(s[5], 1 | 3 | 4) {
                            return Walk::Pillow;
                        }
                        size = (
                            u16::from_be_bytes([s[3], s[4]]),
                            u16::from_be_bytes([s[1], s[2]]),
                        );
                    }
                    0xFFDB => {
                        let mut rest = s;
                        while let Some(&v) = rest.first() {
                            let table = 1 + 64 * if v >> 4 == 0 { 1 } else { 2 };
                            if rest.len() < table {
                                return Walk::Pillow;
                            }
                            rest = &rest[table..];
                        }
                    }
                    0xFFE0..=0xFFEF => {
                        let app = marker & 0xF;
                        if app == 0 && s.starts_with(b"JFIF") && s.len() < 7 {
                            return Walk::Pillow;
                        }
                        if app == 1 && s.starts_with(b"Exif\0\0") {
                            if exif.is_some() {
                                return Walk::Pillow; // Pillow concatenates; not replicated
                            }
                            exif = Some(s);
                        }
                        if app == 1 && s.starts_with(b"http://ns.adobe.com/xap/1.0/\x00") {
                            xmp = true;
                        }
                        if app == 2
                            && (s.starts_with(b"MPF\0")
                                || (s.starts_with(b"ICC_PROFILE\0") && s.len() < 14))
                        {
                            return Walk::Pillow;
                        }
                        if app == 13 && s.starts_with(b"Photoshop 3.0\x00") && !photoshop_parses(s)
                        {
                            return Walk::Pillow;
                        }
                        if app == 14 && s.starts_with(b"Adobe") && s.len() < 7 {
                            return Walk::Pillow;
                        }
                    }
                    _ => {} // DHT, DAC, SOS, DNL, DRI, EXP, COM: skipped segments
                }
                if marker == 0xFFDA {
                    break;
                }
            }
            _ => return Walk::Pillow, // "no marker found"
        }
        need!(pos + 1);
        current = buf[pos];
        pos += 1;
    }
    if sof_count != 1 {
        return Walk::Pillow;
    }
    let orientation = match exif.map(exif_orientation) {
        None => None,
        Some(Ok(value)) => value,
        Some(Err(())) => return Walk::Pillow,
    };
    if orientation.is_none() && xmp {
        return Walk::Pillow; // Pillow may take Orientation from XMP
    }
    Walk::Done {
        width: size.0,
        height: size.1,
        orientation,
    }
}

#[cfg(unix)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match file.read_at(&mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

#[cfg(windows)]
fn read_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<usize> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        match file.seek_read(&mut buf[done..], offset + done as u64) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// (status, height, width) with the reference's EXIF-corrected (h, w).
fn probe(path: &str, buf: &mut Vec<u8>, max_pixels: u64, max_bytes: u64) -> (u8, u32, u32) {
    const REJECT: (u8, u32, u32) = (PILLOW, 0, 0);
    let Ok(file) = File::open(path) else {
        return REJECT;
    };
    let Ok(meta) = file.metadata() else {
        return REJECT;
    };
    let total = meta.len();
    if !meta.is_file() || total < 4 || total > max_bytes {
        return REJECT;
    }
    let mut want = FIRST_READ.min(total as usize);
    let (width, height, orientation) = loop {
        buf.resize(want, 0);
        match read_at(&file, buf, 0) {
            Ok(n) if n == want => {}
            _ => return REJECT,
        }
        match walk(buf, want as u64 == total) {
            Walk::Done {
                width,
                height,
                orientation,
            } => break (width, height, orientation),
            Walk::Pillow => return REJECT,
            Walk::NeedMore => want = (want * 4).min(total as usize),
        }
    };
    let mut tail = [0u8; 2];
    if buf.len() as u64 == total {
        tail.copy_from_slice(&buf[buf.len() - 2..]);
    } else if !matches!(read_at(&file, &mut tail, total - 2), Ok(2)) {
        return REJECT;
    }
    if tail != [0xFF, 0xD9] {
        return REJECT; // the reference would repair or reject this JPEG
    }
    let (w, h) = (u32::from(width), u32::from(height));
    if max_pixels > 0 && u64::from(w) * u64::from(h) > max_pixels {
        return REJECT; // DecompressionBomb warning/error path
    }
    // exif_size swaps for rotations 6 and 8; the reference then reports (h, w).
    let (w, h) = if matches!(orientation, Some(6) | Some(8)) {
        (h, w)
    } else {
        (w, h)
    };
    if h <= 9 || w <= 9 {
        return REJECT;
    }
    (ACCEPTED, h, w)
}

/// Probe every path in parallel; returns (statuses, heights, widths).
#[pyfunction]
fn probe_jpegs(
    py: Python<'_>,
    paths: Vec<String>,
    workers: usize,
    max_pixels: u64,
    max_bytes: u64,
) -> PyResult<(Vec<u8>, Vec<u32>, Vec<u32>)> {
    if workers == 0 || workers > 256 {
        return Err(PyValueError::new_err("workers must be in 1..256"));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let results: Vec<(u8, u32, u32)> = py.allow_threads(|| {
        pool.install(|| {
            paths
                .par_iter()
                .map_init(Vec::new, |buf, path| {
                    probe(path, buf, max_pixels, max_bytes)
                })
                .collect()
        })
    });
    let mut out = (
        Vec::with_capacity(results.len()),
        Vec::with_capacity(results.len()),
        Vec::with_capacity(results.len()),
    );
    for (s, h, w) in results {
        out.0.push(s);
        out.1.push(h);
        out.2.push(w);
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(probe_jpegs, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exif_ifd(little: bool, entries: &[(u16, u16, u32, [u8; 4])]) -> Vec<u8> {
        let mut d = b"Exif\0\0".to_vec();
        let p16 = |v: u16| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        let p32 = |v: u32| {
            if little {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            }
        };
        d.extend_from_slice(if little { b"II\x2a\x00" } else { b"MM\x00\x2a" });
        d.extend_from_slice(&p32(8));
        d.extend_from_slice(&p16(entries.len() as u16));
        for (tag, kind, n, value) in entries {
            d.extend_from_slice(&p16(*tag));
            d.extend_from_slice(&p16(*kind));
            d.extend_from_slice(&p32(*n));
            d.extend_from_slice(value);
        }
        d.extend_from_slice(&p32(0));
        d
    }

    #[test]
    fn orientation_both_endians_and_conservative_rejections() {
        for little in [true, false] {
            let v = if little { [6, 0, 0, 0] } else { [0, 6, 0, 0] };
            assert_eq!(
                exif_orientation(&exif_ifd(little, &[(274, 3, 1, v)])),
                Ok(Some(6))
            );
            assert_eq!(
                exif_orientation(&exif_ifd(little, &[(271, 2, 3, *b"ab\0\0")])),
                Ok(None)
            );
            assert_eq!(
                exif_orientation(&exif_ifd(little, &[(274, 4, 1, v)])),
                Err(())
            );
            assert_eq!(
                exif_orientation(&exif_ifd(little, &[(274, 99, 1, v)])),
                Err(())
            );
            assert_eq!(
                exif_orientation(&exif_ifd(little, &[(271, 2, 64, [0, 0, 0, 200])])),
                Err(())
            );
        }
        assert!(photoshop_parses(
            b"Photoshop 3.0\x008BIM\x04\x04\x00\x00\x00\x00\x00\x00"
        ));
        assert!(photoshop_parses(b"Photoshop 3.0\x008BIM\x04"));
        assert!(!photoshop_parses(b"Photoshop 3.0\x008BIM\x04\x04"));
        assert!(photoshop_parses(
            b"Photoshop 3.0\x008BIM\x03\xed\x00\x00\x00\x00\x02\x00\x00"
        ));
        assert_eq!(exif_orientation(b"Exif\0\0"), Ok(None));
        assert_eq!(exif_orientation(b"Exif\0\0II*\0\x08\0\0\0\x05"), Err(()));
    }
}
