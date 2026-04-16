//! Shared constants and small helpers from ISO/IEC 14496-3.
//!
//! Most of these come from §1.6.2 (Sampling Frequency Index) and §1.6.3
//! (Channel Configuration).

/// AAC profile (object_type) constants — ISO/IEC 14496-3 Table 1.17.
pub const AOT_AAC_MAIN: u8 = 1;
pub const AOT_AAC_LC: u8 = 2;
pub const AOT_AAC_SSR: u8 = 3;
pub const AOT_AAC_LTP: u8 = 4;
pub const AOT_SBR: u8 = 5;
pub const AOT_AAC_SCALABLE: u8 = 6;
pub const AOT_PS: u8 = 29;
pub const AOT_ER_AAC_LC: u8 = 17;
pub const AOT_ER_AAC_LD: u8 = 23;

/// Sampling Frequency Index → sample rate (Hz). Table 1.16.
pub const SAMPLE_RATES: [u32; 13] = [
    96_000, 88_200, 64_000, 48_000, 44_100, 32_000, 24_000, 22_050, 16_000, 12_000, 11_025, 8_000,
    7_350,
];

/// Resolve a Sampling Frequency Index into Hz, or `None` if reserved/escape.
pub fn sample_rate(index: u8) -> Option<u32> {
    SAMPLE_RATES.get(index as usize).copied()
}

/// Element types — ISO/IEC 14496-3 Table 4.71.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    /// Single Channel Element (mono).
    Sce = 0,
    /// Channel Pair Element (stereo).
    Cpe = 1,
    /// Coupling Channel Element.
    Cce = 2,
    /// LFE.
    Lfe = 3,
    /// Data Stream Element.
    Dse = 4,
    /// Program Config Element.
    Pce = 5,
    /// Fill Element.
    Fil = 6,
    /// End of frame.
    End = 7,
}

impl ElementType {
    pub fn from_id(id: u32) -> Self {
        match id {
            0 => Self::Sce,
            1 => Self::Cpe,
            2 => Self::Cce,
            3 => Self::Lfe,
            4 => Self::Dse,
            5 => Self::Pce,
            6 => Self::Fil,
            _ => Self::End,
        }
    }
}

/// Window sequence values — ISO/IEC 14496-3 Table 4.65.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WindowSequence {
    #[default]
    OnlyLong = 0,
    LongStart = 1,
    EightShort = 2,
    LongStop = 3,
}

impl WindowSequence {
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::LongStart,
            2 => Self::EightShort,
            3 => Self::LongStop,
            _ => Self::OnlyLong,
        }
    }

    pub fn is_eight_short(self) -> bool {
        matches!(self, Self::EightShort)
    }
}

/// Window shape — ISO/IEC 14496-3 §4.6.11. 0 = sine, 1 = KBD.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WindowShape {
    #[default]
    Sine = 0,
    Kbd = 1,
}

impl WindowShape {
    pub fn from_bit(v: u32) -> Self {
        if v != 0 {
            Self::Kbd
        } else {
            Self::Sine
        }
    }
}
