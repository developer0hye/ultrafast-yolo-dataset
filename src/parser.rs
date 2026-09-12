// SPDX-License-Identifier: AGPL-3.0-only
// Numeric/ordering semantics follow the pinned Ultralytics label verifier.
use std::cmp::Ordering;

#[derive(Debug, Default)]
pub struct Record {
    pub rows: Vec<[f32; 5]>,
    pub segments: Vec<Vec<[f32; 2]>>,
    pub status: u8, // 0 valid, 1 missing, 2 empty, 3 reference required, 4 resource limit
    pub duplicates: usize,
}

impl Record {
    pub fn status(status: u8) -> Self {
        Self {
            status,
            ..Self::default()
        }
    }
}

fn number(s: &str) -> Option<f32> {
    if !s
        .bytes()
        .all(|c| c.is_ascii_digit() || b".+-eE".contains(&c))
    {
        return None;
    }
    // NumPy's string->float32 oracle rounds through binary64. Direct f32 parsing
    // differs on decimal values just beyond a binary32 midpoint.
    let v = s.parse::<f64>().ok()? as f32;
    v.is_finite().then_some(v)
}

fn compare(a: &[f32; 5], b: &[f32; 5]) -> Ordering {
    for i in 0..5 {
        let c = a[i].partial_cmp(&b[i]).expect("finite values validated");
        if c != Ordering::Equal {
            return c;
        }
    }
    Ordering::Equal
}

pub fn parse(text: &[u8], classes: usize, single_cls: bool) -> Record {
    let text = match std::str::from_utf8(text) {
        Ok(s) => s,
        Err(_) => return Record::status(3),
    };
    // Unicode numeric syntax and Python-specific splitlines are delegated until
    // their exact diagnostic and parsing behavior has an oracle fixture.
    if !text.is_ascii()
        || text
            .bytes()
            .any(|c| matches!(c, 11 | 12 | 28 | 29 | 30 | 133))
    {
        return Record::status(3);
    }
    let lines: Vec<Vec<&str>> = text
        .trim()
        .split(['\n', '\r'])
        .filter(|s| !s.is_empty())
        .map(|s| s.split_whitespace().collect())
        .collect();
    if lines.is_empty() {
        return Record::status(2);
    }
    let segment = lines.iter().any(|l| l.len() > 6);
    if segment && lines.iter().any(|l| l.len() == 5) {
        return Record::status(3);
    }
    let mut result = Record::default();
    for tokens in lines {
        if (!segment && tokens.len() != 5)
            || (segment && (tokens.len() < 3 || tokens.len() % 2 == 0))
        {
            return Record::status(3);
        }
        let values: Option<Vec<f32>> = tokens.into_iter().map(number).collect();
        let values = match values {
            Some(v) => v,
            None => return Record::status(3),
        };
        let row = if segment {
            // NumPy's SIMD min/max can select a different sign for equal zeros
            // depending on stride, length and CPU. Use the explicit reference
            // path until that reduction has an exact native implementation.
            if values[1..].iter().any(|v| v.to_bits() == 0x8000_0000) {
                return Record::status(3);
            }
            // Avoid requiring Rust 1.88 solely for the as_chunks convenience API.
            #[allow(clippy::chunks_exact_to_as_chunks)]
            let points: Vec<[f32; 2]> = values[1..].chunks_exact(2).map(|v| [v[0], v[1]]).collect();
            let mut lo = points[0];
            let mut hi = points[0];
            for point in &points[1..] {
                for k in 0..2 {
                    if point[k] <= lo[k] {
                        lo[k] = point[k];
                    }
                    if point[k] >= hi[k] {
                        hi[k] = point[k];
                    }
                }
            }
            let row = [
                values[0],
                (lo[0] + hi[0]) / 2.0,
                (lo[1] + hi[1]) / 2.0,
                hi[0] - lo[0],
                hi[1] - lo[1],
            ];
            result.segments.push(points);
            row
        } else {
            [values[0], values[1], values[2], values[3], values[4]]
        };
        if row.iter().any(|v| !v.is_finite() || *v < -0.01_f32)
            || row[1..].iter().any(|v| *v > 1.01_f32)
            || (!single_cls && row[0] >= classes as f32)
        {
            return Record::status(3);
        }
        result.rows.push(row);
    }
    // np.unique returns lexicographic order and first occurrences, but the
    // reference applies that permutation ONLY when a duplicate exists.
    let mut indices: Vec<usize> = (0..result.rows.len()).collect();
    indices.sort_by(|a, b| compare(&result.rows[*a], &result.rows[*b]).then(a.cmp(b)));
    indices.dedup_by(|a, b| compare(&result.rows[*a], &result.rows[*b]) == Ordering::Equal);
    result.duplicates = result.rows.len() - indices.len();
    if result.duplicates > 0 {
        result.rows = indices.iter().map(|i| result.rows[*i]).collect();
        if segment {
            result.segments = indices
                .iter()
                .map(|i| std::mem::take(&mut result.segments[*i]))
                .collect();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditional_order() {
        let a = b"2 .5 .5 .2 .2\n0 .4 .4 .1 .1";
        assert_eq!(parse(a, 3, false).rows[0][0], 2.0);
        let b = b"2 .5 .5 .2 .2\n0 .4 .4 .1 .1\n2 .5 .5 .2 .2";
        let r = parse(b, 3, false);
        assert_eq!(r.rows[0][0], 0.0);
        assert_eq!(r.duplicates, 1);
    }
    #[test]
    fn midpoint() {
        assert_eq!(
            number("1.0000000596046448").unwrap().to_bits(),
            1f32.to_bits()
        );
    }
    #[test]
    fn finite_and_utf8() {
        for s in [b"nan".as_slice(), b"\xff", b"1e999"] {
            assert_eq!(parse(s, 2, false).status, 3);
        }
    }
}
