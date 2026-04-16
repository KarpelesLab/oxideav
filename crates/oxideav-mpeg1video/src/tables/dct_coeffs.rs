//! Tables B-14 / B-15 — DCT coefficient VLCs.
//!
//! In the MPEG-1 spec these are two closely related tables; libavcodec
//! represents them as a single 111-entry VLC plus escape + EOB markers
//! (`ff_mpeg1_vlc_table` with `ff_mpeg12_run` / `ff_mpeg12_level` providing
//! the symbol meaning).
//!
//! We reproduce the libavcodec data verbatim and adapt it to oxideav's VLC
//! decoder. The caller interprets the EOB codeword differently for the
//! "first coefficient of a non-intra block" case (where `1s` = run=0,
//! level=±1 instead of EOB) — that dispatch is handled at the block level.

use crate::vlc::VlcEntry;

#[derive(Clone, Copy, Debug)]
pub enum DctSym {
    /// Run of zeros followed by a coefficient whose absolute magnitude is
    /// `level_abs`. Sign bit follows in the bitstream.
    RunLevel {
        run: u8,
        level_abs: u16,
    },
    Eob,
    Escape,
    /// Special marker for "first coefficient" decode: the codeword `1` which
    /// is normally EOB. Resolved at the call site by picking first-coeff or
    /// subsequent-coeff semantics.
    EobOrFirstOne,
}

// 111 (run, level) pairs, matching `ff_mpeg12_run` and `ff_mpeg12_level`.
const LEVEL: [u8; 111] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2,
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

const RUN: [u8; 111] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
    3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14,
    15, 15, 16, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
];

// `(code, bits)` for 111 entries + 2 sentinels (escape = 0x1,6; EOB = 0x2,2).
const MPEG1_VLC: [(u32, u8); 113] = [
    (0x3, 2),
    (0x4, 4),
    (0x5, 5),
    (0x6, 7),
    (0x26, 8),
    (0x21, 8),
    (0xa, 10),
    (0x1d, 12),
    (0x18, 12),
    (0x13, 12),
    (0x10, 12),
    (0x1a, 13),
    (0x19, 13),
    (0x18, 13),
    (0x17, 13),
    (0x1f, 14),
    (0x1e, 14),
    (0x1d, 14),
    (0x1c, 14),
    (0x1b, 14),
    (0x1a, 14),
    (0x19, 14),
    (0x18, 14),
    (0x17, 14),
    (0x16, 14),
    (0x15, 14),
    (0x14, 14),
    (0x13, 14),
    (0x12, 14),
    (0x11, 14),
    (0x10, 14),
    (0x18, 15),
    (0x17, 15),
    (0x16, 15),
    (0x15, 15),
    (0x14, 15),
    (0x13, 15),
    (0x12, 15),
    (0x11, 15),
    (0x10, 15),
    (0x3, 3),
    (0x6, 6),
    (0x25, 8),
    (0xc, 10),
    (0x1b, 12),
    (0x16, 13),
    (0x15, 13),
    (0x1f, 15),
    (0x1e, 15),
    (0x1d, 15),
    (0x1c, 15),
    (0x1b, 15),
    (0x1a, 15),
    (0x19, 15),
    (0x13, 16),
    (0x12, 16),
    (0x11, 16),
    (0x10, 16),
    (0x5, 4),
    (0x4, 7),
    (0xb, 10),
    (0x14, 12),
    (0x14, 13),
    (0x7, 5),
    (0x24, 8),
    (0x1c, 12),
    (0x13, 13),
    (0x6, 5),
    (0xf, 10),
    (0x12, 12),
    (0x7, 6),
    (0x9, 10),
    (0x12, 13),
    (0x5, 6),
    (0x1e, 12),
    (0x14, 16),
    (0x4, 6),
    (0x15, 12),
    (0x7, 7),
    (0x11, 12),
    (0x5, 7),
    (0x11, 13),
    (0x27, 8),
    (0x10, 13),
    (0x23, 8),
    (0x1a, 16),
    (0x22, 8),
    (0x19, 16),
    (0x20, 8),
    (0x18, 16),
    (0xe, 10),
    (0x17, 16),
    (0xd, 10),
    (0x16, 16),
    (0x8, 10),
    (0x15, 16),
    (0x1f, 12),
    (0x1a, 12),
    (0x19, 12),
    (0x17, 12),
    (0x16, 12),
    (0x1f, 13),
    (0x1e, 13),
    (0x1d, 13),
    (0x1c, 13),
    (0x1b, 13),
    (0x1f, 16),
    (0x1e, 16),
    (0x1d, 16),
    (0x1c, 16),
    (0x1b, 16),
    (0x1, 6), // escape
    (0x2, 2), // EOB
];

pub fn table() -> &'static [VlcEntry<DctSym>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<DctSym>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut v = Vec::with_capacity(113);
        for i in 0..111 {
            let (code, bits) = MPEG1_VLC[i];
            v.push(VlcEntry::new(
                bits,
                code,
                DctSym::RunLevel {
                    run: RUN[i],
                    level_abs: LEVEL[i] as u16,
                },
            ));
        }
        // Escape.
        let (code, bits) = MPEG1_VLC[111];
        v.push(VlcEntry::new(bits, code, DctSym::Escape));
        // EOB.
        let (code, bits) = MPEG1_VLC[112];
        v.push(VlcEntry::new(bits, code, DctSym::Eob));
        v
    })
    .as_slice()
}

/// Same table but the first entry (code `1`, 2 bits — normally the `11s`
/// pattern splits into run=0,level=1) is mapped from `RunLevel(0,1)` to a
/// special `EobOrFirstOne` marker that callers must not use at first
/// position (instead calling `decode_first_coeff`).
#[cfg(test)]
mod tests_counts {
    use super::*;

    #[test]
    fn table_sizes() {
        assert_eq!(LEVEL.len(), 111);
        assert_eq!(RUN.len(), 111);
        assert_eq!(MPEG1_VLC.len(), 113);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vlc::VlcEntry;

    fn check_no_collision<T: Copy>(t: &[VlcEntry<T>], name: &str) {
        for (i, e) in t.iter().enumerate() {
            for (j, f) in t.iter().enumerate() {
                if i >= j {
                    continue;
                }
                if e.bits <= f.bits {
                    let f_prefix = f.code >> (f.bits - e.bits) as u32;
                    if f_prefix == e.code {
                        panic!(
                            "{name}: prefix collision: entry {i} (bits={}, code=0x{:x}) is a prefix of entry {j} (bits={}, code=0x{:x})",
                            e.bits, e.code, f.bits, f.code
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn no_prefix_collisions_dct() {
        check_no_collision(table(), "dct_coeffs");
    }

    #[test]
    fn dct_table_spot_checks() {
        // Entry 0: code 0x3 (2 bits) → (run=0, level=1)
        let e0 = &table()[0];
        assert_eq!(e0.code, 0x3);
        assert_eq!(e0.bits, 2);
        if let DctSym::RunLevel { run, level_abs } = e0.value {
            assert_eq!(run, 0);
            assert_eq!(level_abs, 1);
        } else {
            panic!("entry 0 should be RunLevel");
        }
        // Entry 40: code 0x3 (3 bits) → (run=1, level=1)
        let e40 = &table()[40];
        assert_eq!(e40.code, 0x3);
        assert_eq!(e40.bits, 3);
        if let DctSym::RunLevel { run, level_abs } = e40.value {
            assert_eq!(run, 1);
            assert_eq!(level_abs, 1);
        }
        // EOB is at index 112.
        assert!(matches!(table()[112].value, DctSym::Eob));
        assert_eq!(table()[112].code, 0x2);
        assert_eq!(table()[112].bits, 2);
        // Escape at index 111.
        assert!(matches!(table()[111].value, DctSym::Escape));
        assert_eq!(table()[111].code, 0x1);
        assert_eq!(table()[111].bits, 6);
    }

    #[test]
    fn no_prefix_collisions_tables() {
        check_no_collision(crate::tables::mba::table(), "mba");
        check_no_collision(crate::tables::cbp::table(), "cbp");
        check_no_collision(crate::tables::motion::table(), "motion");
        check_no_collision(crate::tables::dct_dc::luma(), "dct_dc_luma");
        check_no_collision(crate::tables::dct_dc::chroma(), "dct_dc_chroma");
    }
}

pub fn first_coeff_table() -> &'static [VlcEntry<DctSym>] {
    use std::sync::OnceLock;
    static CELL: OnceLock<Vec<VlcEntry<DctSym>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut v = Vec::with_capacity(113);
        for i in 0..111 {
            let (code, bits) = MPEG1_VLC[i];
            if i == 0 {
                // The "1s" (2-bit code `1` + sign) decoded at first position
                // means run=0, level=±1 — not EOB.
                v.push(VlcEntry::new(
                    bits,
                    code,
                    DctSym::RunLevel {
                        run: 0,
                        level_abs: 1,
                    },
                ));
                continue;
            }
            v.push(VlcEntry::new(
                bits,
                code,
                DctSym::RunLevel {
                    run: RUN[i],
                    level_abs: LEVEL[i] as u16,
                },
            ));
        }
        // Escape.
        let (code, bits) = MPEG1_VLC[111];
        v.push(VlcEntry::new(bits, code, DctSym::Escape));
        // EOB is NOT a valid first coefficient — but include it anyway for
        // robustness; real encoders never emit EOB at first position.
        let (code, bits) = MPEG1_VLC[112];
        v.push(VlcEntry::new(bits, code, DctSym::Eob));
        v
    })
    .as_slice()
}
