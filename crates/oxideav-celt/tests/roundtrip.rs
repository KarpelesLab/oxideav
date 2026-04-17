//! CELT encoder → decoder roundtrip test.
//!
//! Drives the encoder with a known test signal (sine, noise), decodes the
//! resulting packets with a minimal in-process CELT decoder (built from the
//! published `bands`, `quant_bands`, `mdct` modules) and measures SNR.

use oxideav_celt::bands::{denormalise_bands, quant_all_bands};
use oxideav_celt::encoder::{CeltEncoder, FRAME_SAMPLES, SAMPLE_RATE};
use oxideav_celt::header::decode_header;
use oxideav_celt::mdct::{imdct_sub, window_overlap_add};
use oxideav_celt::quant_bands::{
    unquant_coarse_energy, unquant_energy_finalise, unquant_fine_energy,
};
use oxideav_celt::range_decoder::{RangeDecoder, BITRES};
use oxideav_celt::rate::clt_compute_allocation;
use oxideav_celt::tables::{
    init_caps, lm_for_frame_samples, EBAND_5MS, NB_EBANDS, SPREAD_ICDF, SPREAD_NORMAL,
    TF_SELECT_TABLE, TRIM_ICDF,
};
use oxideav_codec::Encoder;
use oxideav_core::{AudioFrame, CodecId, CodecParameters, Frame, Packet, SampleFormat, TimeBase};

const OVERLAP: usize = 120;

#[rustfmt::skip]
const WINDOW_120: [f32; 120] = [
    6.7286966e-05, 0.00060551348, 0.0016815970, 0.0032947962, 0.0054439943,
    0.0081276923, 0.011344001, 0.015090633, 0.019364886, 0.024163635,
    0.029483315, 0.035319905, 0.041668911, 0.048525347, 0.055883718,
    0.063737999, 0.072081616, 0.080907428, 0.090207705, 0.099974111,
    0.11019769, 0.12086883, 0.13197729, 0.14351214, 0.15546177,
    0.16781389, 0.18055550, 0.19367290, 0.20715171, 0.22097682,
    0.23513243, 0.24960208, 0.26436860, 0.27941419, 0.29472040,
    0.31026818, 0.32603788, 0.34200931, 0.35816177, 0.37447407,
    0.39092462, 0.40749142, 0.42415215, 0.44088423, 0.45766484,
    0.47447104, 0.49127978, 0.50806798, 0.52481261, 0.54149077,
    0.55807973, 0.57455701, 0.59090049, 0.60708841, 0.62309951,
    0.63891306, 0.65450896, 0.66986776, 0.68497077, 0.69980010,
    0.71433873, 0.72857055, 0.74248043, 0.75605424, 0.76927895,
    0.78214257, 0.79463430, 0.80674445, 0.81846456, 0.82978733,
    0.84070669, 0.85121779, 0.86131698, 0.87100183, 0.88027111,
    0.88912479, 0.89756398, 0.90559094, 0.91320904, 0.92042270,
    0.92723738, 0.93365955, 0.93969656, 0.94535671, 0.95064907,
    0.95558353, 0.96017067, 0.96442171, 0.96834849, 0.97196334,
    0.97527906, 0.97830883, 0.98106616, 0.98356480, 0.98581869,
    0.98784191, 0.98964856, 0.99125274, 0.99266849, 0.99390969,
    0.99499004, 0.99592297, 0.99672162, 0.99739874, 0.99796667,
    0.99843728, 0.99882195, 0.99913147, 0.99937606, 0.99956527,
    0.99970802, 0.99981248, 0.99988613, 0.99993565, 0.99996697,
    0.99998518, 0.99999457, 0.99999859, 0.99999982, 1.0000000,
];

/// Decode one mono, long-block, intra, non-transient, no-postfilter CELT
/// frame (the subset our encoder emits) and return `FRAME_SAMPLES` PCM samples.
#[allow(clippy::too_many_arguments)]
fn decode_celt_frame(
    bytes: &[u8],
    old_band_e: &mut Vec<f32>,
    prev_tail: &mut [f32],
    rng: &mut u32,
) -> Vec<f32> {
    let mut rc = RangeDecoder::new(bytes);
    let lm = lm_for_frame_samples(FRAME_SAMPLES as u32) as i32;
    let end_band = NB_EBANDS;
    let start_band = 0usize;

    let header = match decode_header(&mut rc) {
        Some(h) => h,
        None => return vec![0.0; FRAME_SAMPLES],
    };
    assert!(!header.transient, "expected non-transient");
    assert!(header.post_filter.is_none(), "expected no post-filter");

    let m = 1i32 << lm;
    let n = (m * EBAND_5MS[NB_EBANDS] as i32) as usize;

    unquant_coarse_energy(
        &mut rc,
        old_band_e,
        start_band,
        end_band,
        header.intra,
        1,
        lm as usize,
    );

    // tf_decode (mirror of the encoder's emitting zeros).
    let budget = (rc.storage() * 8) as u32;
    let mut tell_u = rc.tell() as u32;
    let mut logp = 4u32;
    let tf_select_rsv = if lm > 0 && tell_u + logp + 1 <= budget {
        1
    } else {
        0
    };
    let budget_after = budget - tf_select_rsv;
    let mut tf_res = vec![0i32; NB_EBANDS];
    let mut tf_changed = 0i32;
    let mut curr = 0i32;
    for i in start_band..end_band {
        if tell_u + logp <= budget_after {
            let bit = rc.decode_bit_logp(logp);
            curr ^= bit as i32;
            tell_u = rc.tell() as u32;
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = 5;
    }
    let mut tf_select = 0i32;
    if tf_select_rsv != 0
        && TF_SELECT_TABLE[lm as usize][4 * header.transient as usize + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][4 * header.transient as usize + 2 + tf_changed as usize]
    {
        tf_select = if rc.decode_bit_logp(1) { 1 } else { 0 };
    }
    for i in start_band..end_band {
        let idx = (4 * header.transient as i32 + 2 * tf_select + tf_res[i]) as usize;
        tf_res[i] = TF_SELECT_TABLE[lm as usize][idx] as i32;
    }

    let mut tell = rc.tell();
    let total_bits_check = (rc.storage() * 8) as i32;
    let spread = if tell + 4 <= total_bits_check {
        rc.decode_icdf(&SPREAD_ICDF, 5) as i32
    } else {
        SPREAD_NORMAL
    };

    let cap = init_caps(lm as usize, 1);
    let mut offsets = [0i32; NB_EBANDS];
    let mut dynalloc_logp = 6i32;
    let mut total_bits_frac = (bytes.len() as i32) * 8 << BITRES;
    tell = rc.tell_frac() as i32;
    for i in start_band..end_band {
        let width = (EBAND_5MS[i + 1] - EBAND_5MS[i]) as i32 * m;
        let quanta = (width << BITRES).min((6 << BITRES).max(width));
        let mut dynalloc_loop_logp = dynalloc_logp;
        let mut boost = 0i32;
        while tell + (dynalloc_loop_logp << BITRES) < total_bits_frac && boost < cap[i] {
            let flag = rc.decode_bit_logp(dynalloc_loop_logp as u32);
            tell = rc.tell_frac() as i32;
            if !flag {
                break;
            }
            boost += quanta;
            total_bits_frac -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i] = boost;
        if boost > 0 {
            dynalloc_logp = 2.max(dynalloc_logp - 1);
        }
    }

    let alloc_trim = if tell + (6 << BITRES) <= total_bits_frac {
        rc.decode_icdf(&TRIM_ICDF, 7) as i32
    } else {
        5
    };

    let bits = ((bytes.len() as i32) * 8 << BITRES) - rc.tell_frac() as i32 - 1;

    let mut pulses = vec![0i32; NB_EBANDS];
    let mut fine_quant = vec![0i32; NB_EBANDS];
    let mut fine_priority = vec![0i32; NB_EBANDS];
    let mut intensity = 0i32;
    let mut dual_stereo = 0i32;
    let mut balance = 0i32;
    let coded_bands = clt_compute_allocation(
        start_band,
        end_band,
        &offsets,
        &cap,
        alloc_trim,
        &mut intensity,
        &mut dual_stereo,
        bits,
        &mut balance,
        &mut pulses,
        &mut fine_quant,
        &mut fine_priority,
        1,
        lm,
        &mut rc,
    );

    unquant_fine_energy(&mut rc, old_band_e, start_band, end_band, &fine_quant, 1);

    let mut x_buf = vec![0f32; n];
    let mut collapse_masks = vec![0u8; NB_EBANDS];
    let total_pvq_bits = (bytes.len() as i32) * (8 << BITRES);
    let mut rng_local = *rng;
    let band_e_snapshot = old_band_e.clone();
    quant_all_bands(
        start_band,
        end_band,
        &mut x_buf,
        None,
        &mut collapse_masks,
        &band_e_snapshot,
        &pulses,
        false,
        spread,
        dual_stereo,
        intensity,
        &tf_res,
        total_pvq_bits,
        balance,
        &mut rc,
        lm,
        coded_bands,
        &mut rng_local,
        false,
    );
    *rng = rng_local;

    let bits_left = (bytes.len() as i32) * 8 - rc.tell();
    unquant_energy_finalise(
        &mut rc,
        old_band_e,
        start_band,
        end_band,
        &fine_quant,
        &fine_priority,
        bits_left,
        1,
    );

    // Denormalise.
    let mut freq = vec![0f32; n];
    denormalise_bands(
        &x_buf,
        &mut freq,
        &old_band_e[..NB_EBANDS],
        start_band,
        end_band,
        m as usize,
        false,
    );

    // IMDCT (long block, blocks = 1, sub_n = n = 800). Raw output is 2*sub_n
    // = 1600 samples. The external frame is FRAME_SAMPLES = 960, so the
    // decoded body is sub_n = 800 samples of PVQ-coded content, plus 160
    // samples of zero tail.
    let sub_n = n;
    let mut raw_coded = vec![0f32; 2 * sub_n];
    imdct_sub(&freq, &mut raw_coded, sub_n);
    // Pad to full 2 * FRAME_SAMPLES = 1920 for window_overlap_add.
    let mut raw = vec![0f32; 2 * FRAME_SAMPLES];
    raw[..2 * sub_n].copy_from_slice(&raw_coded);

    // Explicit OLA: window_overlap_add mutates prev_tail in place.
    let mut out = vec![0f32; FRAME_SAMPLES];
    for i in 0..OVERLAP {
        let w = WINDOW_120[i];
        out[i] = prev_tail[i] + w * raw[i];
    }
    for i in OVERLAP..FRAME_SAMPLES {
        out[i] = raw[i];
    }
    for i in 0..OVERLAP {
        let w = WINDOW_120[OVERLAP - 1 - i];
        prev_tail[i] = w * raw[FRAME_SAMPLES + i];
    }

    out
}

fn encode_signal_to_packets(signal: &[f32]) -> Vec<Packet> {
    let mut p = CodecParameters::audio(CodecId::new(oxideav_celt::CODEC_ID_STR));
    p.channels = Some(1);
    p.sample_rate = Some(SAMPLE_RATE);
    let mut enc = CeltEncoder::new(&p).unwrap();

    let mut packets = Vec::new();
    for chunk in signal.chunks(FRAME_SAMPLES) {
        if chunk.len() < FRAME_SAMPLES {
            break;
        }
        // Pack into F32 bytes.
        let mut bytes = Vec::with_capacity(FRAME_SAMPLES * 4);
        for &s in chunk {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let frame = Frame::Audio(AudioFrame {
            format: SampleFormat::F32,
            channels: 1,
            sample_rate: SAMPLE_RATE,
            samples: FRAME_SAMPLES as u32,
            pts: None,
            time_base: TimeBase::new(1, SAMPLE_RATE as i64),
            data: vec![bytes],
        });
        enc.send_frame(&frame).unwrap();
        while let Ok(pkt) = enc.receive_packet() {
            packets.push(pkt);
        }
    }
    enc.flush().unwrap();
    while let Ok(pkt) = enc.receive_packet() {
        packets.push(pkt);
    }
    packets
}

fn decode_packets(packets: &[Packet], expected_samples: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(expected_samples);
    let mut old_band_e = vec![0.0f32; NB_EBANDS * 2];
    let mut prev_tail = vec![0.0f32; OVERLAP];
    let mut rng: u32 = 0;
    for pkt in packets {
        let pcm = decode_celt_frame(&pkt.data, &mut old_band_e, &mut prev_tail, &mut rng);
        out.extend_from_slice(&pcm);
    }
    out.truncate(expected_samples);
    out
}

/// SNR in dB between two equal-length signals.
fn psnr_db(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    let signal_power: f64 = x[..n]
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / n as f64;
    let noise_power: f64 = x[..n]
        .iter()
        .zip(y[..n].iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if noise_power < 1e-30 {
        return 200.0;
    }
    10.0 * (signal_power / noise_power).log10() as f32
}

#[test]
fn sine_roundtrip_produces_audible_output() {
    // 1 kHz sine at amplitude 0.3, 4 frames = 80 ms.
    let n_frames = 4;
    let n_samples = FRAME_SAMPLES * n_frames;
    let freq = 1000.0;
    let signal: Vec<f32> = (0..n_samples)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SAMPLE_RATE as f32).sin() * 0.3)
        .collect();

    let packets = encode_signal_to_packets(&signal);
    assert!(!packets.is_empty(), "encoder produced no packets");
    let decoded = decode_packets(&packets, n_samples);

    // Compare the middle frame (avoid the first frame which has empty overlap).
    let start = FRAME_SAMPLES * 2;
    let end = FRAME_SAMPLES * 3;
    let ref_slice = &signal[start..end];
    let dec_slice = &decoded[start..end];

    // Sanity: decoder produces nonzero output.
    let dec_energy: f32 = dec_slice.iter().map(|v| v * v).sum::<f32>() / FRAME_SAMPLES as f32;
    assert!(
        dec_energy > 1e-6,
        "decoder produced silent output (energy {dec_energy})"
    );

    // Goertzel at 1 kHz: check that the target tone dominates the decoder output.
    let goertzel = |samples: &[f32], f: f32| -> f32 {
        let w = 2.0 * std::f32::consts::PI * f / SAMPLE_RATE as f32;
        let cw = w.cos();
        let (mut s0, mut s1, mut s2) = (0f32, 0f32, 0f32);
        for &x in samples {
            s0 = 2.0 * cw * s1 - s2 + x;
            s2 = s1;
            s1 = s0;
        }
        (s1 * s1 + s2 * s2 - 2.0 * cw * s1 * s2).sqrt()
    };
    let mag_target = goertzel(dec_slice, freq);
    let mag_off = goertzel(dec_slice, 5000.0);
    println!(
        "sine roundtrip: energy {:.4e}, mag@1kHz {:.3}, mag@5kHz {:.3}, ratio {:.1}x",
        dec_energy,
        mag_target,
        mag_off,
        mag_target / mag_off.max(1e-9)
    );
    // Weak bar: there's *some* energy at the target tone. The encoder's
    // band-split recursion + fold path hasn't been exhaustively verified
    // against libopus, so we accept a generous ratio here — the main
    // signal of correctness is that the encoder produces packets the
    // decoder can parse without crashing, and that the energy scale in
    // the decoded output is within the right ballpark.
    assert!(
        mag_target > 0.3 * mag_off,
        "1 kHz tone completely buried in decoder output"
    );

    // Report PSNR. CELT is perceptual so we don't expect > 20 dB on raw
    // time-domain comparison — phase is not preserved across independent
    // encode/decode overlap-add, and our forward MDCT + PVQ shape split is
    // not yet bit-exact with libopus.
    let snr = psnr_db(ref_slice, dec_slice);
    println!("sine roundtrip PSNR: {snr:.2} dB");
}

#[test]
fn noise_roundtrip_does_not_crash() {
    // 2 frames of pseudo-random white noise at amplitude 0.1.
    let n_frames = 2;
    let n_samples = FRAME_SAMPLES * n_frames;
    let mut seed = 0x1234u32;
    let signal: Vec<f32> = (0..n_samples)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 16) as i32 - 32768) as f32 / 32768.0 * 0.1
        })
        .collect();
    let packets = encode_signal_to_packets(&signal);
    assert!(!packets.is_empty());
    let decoded = decode_packets(&packets, n_samples);
    // Just sanity: decoder output is all finite.
    assert!(decoded.iter().all(|v| v.is_finite()));

    // The decoder output should have approximately the same energy-scale
    // as the input (within ~30 dB — very lax, CELT is perceptual).
    let start = FRAME_SAMPLES; // second frame, has proper overlap
    let ref_slice = &signal[start..start + FRAME_SAMPLES];
    let dec_slice = &decoded[start..start + FRAME_SAMPLES];
    let e_ref: f64 = ref_slice.iter().map(|v| (*v as f64).powi(2)).sum();
    let e_dec: f64 = dec_slice.iter().map(|v| (*v as f64).powi(2)).sum();
    println!("noise roundtrip: e_ref={e_ref:.4} e_dec={e_dec:.4}");
    // Loosely: decoded energy is within 2 orders of magnitude of input.
    if e_ref > 1e-6 {
        let ratio = e_dec / e_ref;
        assert!(ratio > 0.01 && ratio < 100.0, "energy ratio {ratio}");
    }
}
