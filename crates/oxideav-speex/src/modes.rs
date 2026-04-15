//! Speex sub-mode descriptors.
//!
//! Speex doesn't transmit a single "bitrate"; each encoded frame starts
//! with a 4-bit sub-mode selector that picks a preset of
//! pitch/innovation/LSP quantisers. There are separate tables for
//! narrowband (NB) and high-band (the extra detail layer added for
//! wideband and ultra-wideband). Only a subset of sub-modes are actually
//! populated in the reference encoder.
//!
//! Tables here are intentionally skeletal — enough to let the decoder
//! identify "does this sub-mode exist" and how many bits the frame
//! should consume, but the actual codebook tables are left as stubs to
//! be filled in when the main synthesis loop lands. See
//! `libspeex/modes.c` and `libspeex/modes_wb.c` in the xiph reference
//! for the full tables.

/// Narrowband sub-mode descriptor.
#[derive(Clone, Copy, Debug)]
pub struct NbSubmode {
    /// Average bitrate for this sub-mode at the nominal 8 kHz rate.
    pub bitrate: u32,
    /// LSP quantisation table selector (0 = unused, otherwise picks a
    /// codebook — see `lsp_tables_nb.c`).
    pub lsp_quant: u8,
    /// LTP (long-term prediction / pitch) parameters selector.
    pub ltp_quant: u8,
    /// Innovation quantisation parameters selector.
    pub innov_quant: u8,
    /// Number of bits this sub-mode draws per frame from the bitstream.
    pub frame_bits: u32,
}

/// Narrowband sub-modes 0..=8. Sub-mode 0 ("silence") is valid on the
/// wire but transmits no parameters.
pub const NB_SUBMODES: [Option<NbSubmode>; 9] = [
    // 0: silence / discontinuous transmission
    Some(NbSubmode {
        bitrate: 250,
        lsp_quant: 0,
        ltp_quant: 0,
        innov_quant: 0,
        frame_bits: 5,
    }),
    // 1: 5.95 kbit/s (vocoder)
    Some(NbSubmode {
        bitrate: 5_950,
        lsp_quant: 1,
        ltp_quant: 1,
        innov_quant: 1,
        frame_bits: 43,
    }),
    // 2: 8 kbit/s
    Some(NbSubmode {
        bitrate: 8_000,
        lsp_quant: 1,
        ltp_quant: 2,
        innov_quant: 2,
        frame_bits: 119,
    }),
    // 3: 11 kbit/s
    Some(NbSubmode {
        bitrate: 11_000,
        lsp_quant: 1,
        ltp_quant: 3,
        innov_quant: 3,
        frame_bits: 160,
    }),
    // 4: 15 kbit/s
    Some(NbSubmode {
        bitrate: 15_000,
        lsp_quant: 1,
        ltp_quant: 3,
        innov_quant: 4,
        frame_bits: 220,
    }),
    // 5: 18.2 kbit/s
    Some(NbSubmode {
        bitrate: 18_200,
        lsp_quant: 1,
        ltp_quant: 3,
        innov_quant: 5,
        frame_bits: 300,
    }),
    // 6: 24.6 kbit/s
    Some(NbSubmode {
        bitrate: 24_600,
        lsp_quant: 1,
        ltp_quant: 3,
        innov_quant: 6,
        frame_bits: 364,
    }),
    // 7: 3.95 kbit/s (very low bitrate)
    Some(NbSubmode {
        bitrate: 3_950,
        lsp_quant: 1,
        ltp_quant: 4,
        innov_quant: 7,
        frame_bits: 79,
    }),
    // 8: unused in the reference encoder, reserved
    None,
];

/// High-band (wideband layer) sub-mode descriptor. Applied on top of the
/// narrowband frame for modes 1 and 2.
#[derive(Clone, Copy, Debug)]
pub struct HbSubmode {
    pub bitrate: u32,
    /// Number of bits this sub-mode adds to the frame.
    pub frame_bits: u32,
}

/// High-band sub-modes 0..=4 (used for wideband + ultra-wideband layers).
pub const HB_SUBMODES: [Option<HbSubmode>; 5] = [
    // 0: silence — no extra bits besides the sub-mode id
    Some(HbSubmode {
        bitrate: 0,
        frame_bits: 4,
    }),
    // 1: ~5 kbit/s high-band detail
    Some(HbSubmode {
        bitrate: 5_000,
        frame_bits: 36,
    }),
    // 2: ~10 kbit/s
    Some(HbSubmode {
        bitrate: 10_000,
        frame_bits: 112,
    }),
    // 3: ~14 kbit/s
    Some(HbSubmode {
        bitrate: 14_000,
        frame_bits: 192,
    }),
    // 4: ~18 kbit/s
    Some(HbSubmode {
        bitrate: 18_000,
        frame_bits: 352,
    }),
];

/// Look up a narrowband sub-mode. Returns `None` for reserved slots.
pub fn nb_submode(id: u32) -> Option<&'static NbSubmode> {
    if (id as usize) < NB_SUBMODES.len() {
        NB_SUBMODES[id as usize].as_ref()
    } else {
        None
    }
}

/// Look up a high-band (wideband) sub-mode.
pub fn hb_submode(id: u32) -> Option<&'static HbSubmode> {
    if (id as usize) < HB_SUBMODES.len() {
        HB_SUBMODES[id as usize].as_ref()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nb_submodes_cover_0_to_7() {
        for id in 0..=7u32 {
            assert!(nb_submode(id).is_some(), "nb submode {id} should exist");
        }
        assert!(nb_submode(8).is_none(), "nb submode 8 is reserved");
    }

    #[test]
    fn hb_submodes_cover_0_to_4() {
        for id in 0..=4u32 {
            assert!(hb_submode(id).is_some(), "hb submode {id} should exist");
        }
        assert!(hb_submode(5).is_none());
    }

    #[test]
    fn nb_silence_submode_is_tiny() {
        let sm = nb_submode(0).unwrap();
        assert!(sm.frame_bits < 16);
    }
}
