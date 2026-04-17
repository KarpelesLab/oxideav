//! Smoke tests for AV1 intra-prediction + inverse-transform primitives.
//!
//! A fully handcrafted AV1 bitstream that produces a decodable
//! `receive_frame()` output is out of scope for this milestone — it
//! requires default CDF tables + per-block mode / coefficient decode,
//! which still return `Error::Unsupported`. What's in scope is an
//! end-to-end test of the primitives that *will* be driven by that
//! syntax once it lands: DC_PRED neighbour averaging, V_PRED / H_PRED
//! copy patterns, and a DCT round-trip where "no coefficients" is a
//! no-op.
//!
//! Together these lock in the mathematical contract of the pieces that
//! will reconstruct the first real I-frame pixel output.

use oxideav_av1::intra::{predict, IntraMode, Neighbours};
use oxideav_av1::transform::{inverse_transform_add, TxType};
use oxideav_core::Error;

#[test]
fn dc_pred_solid_patch_matches_spec_average() {
    // A 4x4 block surrounded by a single uniform value must predict that
    // value exactly — the canonical case for the "solid colour" I-frame.
    let above = [200u8; 4];
    let left = [200u8; 4];
    let n = Neighbours {
        above: Some(&above),
        left: Some(&left),
    };
    let mut dst = [0u8; 16];
    predict(IntraMode::Dc, n, 4, 4, &mut dst, 4).unwrap();
    for &v in &dst {
        assert_eq!(v, 200, "DC_PRED must collapse a uniform border to itself");
    }
}

#[test]
fn dc_pred_asymmetric_neighbours_round_to_spec() {
    // Above row = 0, Left col = 255. Mean = (0*4 + 255*4) / 8 = 127.5
    // Per AV1 spec rounding (+ denom/2 before divide) → 128.
    let above = [0u8; 4];
    let left = [255u8; 4];
    let n = Neighbours {
        above: Some(&above),
        left: Some(&left),
    };
    let mut dst = [0u8; 16];
    predict(IntraMode::Dc, n, 4, 4, &mut dst, 4).unwrap();
    for &v in &dst {
        assert_eq!(v, 128, "asymmetric neighbours round per AV1 DC formula");
    }
}

#[test]
fn v_pred_produces_vertical_stripes() {
    let above = [10u8, 50, 90, 130];
    let n = Neighbours {
        above: Some(&above),
        left: None,
    };
    let mut dst = [0u8; 16];
    predict(IntraMode::V, n, 4, 4, &mut dst, 4).unwrap();
    for row in 0..4 {
        for c in 0..4 {
            assert_eq!(dst[row * 4 + c], above[c]);
        }
    }
}

#[test]
fn h_pred_produces_horizontal_stripes() {
    let left = [10u8, 50, 90, 130];
    let n = Neighbours {
        above: None,
        left: Some(&left),
    };
    let mut dst = [0u8; 16];
    predict(IntraMode::H, n, 4, 4, &mut dst, 4).unwrap();
    for row in 0..4 {
        for c in 0..4 {
            assert_eq!(dst[row * 4 + c], left[row]);
        }
    }
}

#[test]
fn inverse_dct_zero_residual_preserves_predictor() {
    // After a solid-colour DC_PRED fills a 4x4 block, adding a zero
    // residual via the inverse DCT must be a no-op. This is the tight
    // contract for the "flat" I-frame output path.
    let above = [77u8; 4];
    let left = [77u8; 4];
    let n = Neighbours {
        above: Some(&above),
        left: Some(&left),
    };
    let mut dst = [0u8; 16];
    predict(IntraMode::Dc, n, 4, 4, &mut dst, 4).unwrap();
    let zero_coeffs = [0i32; 16];
    inverse_transform_add(TxType::DctDct, 4, 4, &zero_coeffs, &mut dst, 4).unwrap();
    for &v in &dst {
        assert_eq!(
            v, 77,
            "zero-coefficient iDCT must leave predictor untouched"
        );
    }
}

#[test]
fn inverse_dct_8x8_zero_residual_preserves_predictor() {
    let above = [64u8; 8];
    let left = [64u8; 8];
    let n = Neighbours {
        above: Some(&above),
        left: Some(&left),
    };
    let mut dst = [0u8; 64];
    predict(IntraMode::Dc, n, 8, 8, &mut dst, 8).unwrap();
    let zero_coeffs = [0i32; 64];
    inverse_transform_add(TxType::DctDct, 8, 8, &zero_coeffs, &mut dst, 8).unwrap();
    for &v in &dst {
        assert_eq!(v, 64);
    }
}

#[test]
fn full_solid_colour_path_yields_uniform_block() {
    // The "minimum viable I-frame" path: DC_PRED from neighbours +
    // zero-coefficient iDCT. A real decoder walks the partition tree
    // and drives this pipeline per leaf block. This test locks in the
    // contract for the simplest possible leaf decode.
    let colour: u8 = 175;
    let above = [colour; 4];
    let left = [colour; 4];
    let n = Neighbours {
        above: Some(&above),
        left: Some(&left),
    };
    let mut block = [0u8; 16];
    predict(IntraMode::Dc, n, 4, 4, &mut block, 4).unwrap();
    let zero = [0i32; 16];
    inverse_transform_add(TxType::DctDct, 4, 4, &zero, &mut block, 4).unwrap();
    for &v in &block {
        assert_eq!(
            v, colour,
            "solid-colour I-frame pipeline produced wrong sample"
        );
    }
}

#[test]
fn unsupported_intra_modes_have_precise_error_text() {
    let n = Neighbours {
        above: None,
        left: None,
    };
    let mut dst = [0u8; 16];
    for m in [
        IntraMode::Paeth,
        IntraMode::Smooth,
        IntraMode::D45,
        IntraMode::D135,
    ] {
        match predict(m, n, 4, 4, &mut dst, 4) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains(m.name()), "msg should name mode: {s}");
                assert!(s.contains("§7.11.2"), "msg should ref §7.11.2: {s}");
            }
            other => panic!("expected Unsupported for {}, got {:?}", m.name(), other),
        }
    }
}

#[test]
fn unsupported_transform_sizes_have_precise_error_text() {
    let coeffs = vec![0i32; 16 * 16];
    let mut dst = vec![0u8; 16 * 16];
    match inverse_transform_add(TxType::DctDct, 16, 16, &coeffs, &mut dst, 16) {
        Err(Error::Unsupported(s)) => {
            assert!(s.contains("16"), "msg should name size: {s}");
            assert!(s.contains("§7.7"), "msg should ref §7.7: {s}");
        }
        other => panic!("expected Unsupported for 16×16 iDCT, got {other:?}"),
    }
}
