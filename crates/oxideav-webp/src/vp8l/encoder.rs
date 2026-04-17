//! VP8L lossless encoder.
//!
//! First-cut pure-Rust VP8L encoder. The output is a valid VP8L bitstream
//! (decodable by the in-crate [`super::decode`]), but intentionally
//! unoptimised compared to libwebp:
//!
//! * **No transforms.** Predictor / colour / subtract-green / colour-
//!   indexing are scoped non-goals for this first cut. Files therefore
//!   carry every pixel's raw ARGB value through the entropy coder, which
//!   hurts ratio but keeps the encoder simple. The `0` transform-present
//!   flag is written directly after the header.
//! * **No colour cache.** Cache codes would bloat the green alphabet and
//!   complicate the match search; we skip them.
//! * **No meta-Huffman image.** A single Huffman group covers the whole
//!   picture.
//!
//! What *is* implemented:
//!
//! * Length-limited canonical Huffman tree builder (≤15 bits per code,
//!   matching the VP8L spec's §5 limit) using a frequency-driven sort +
//!   depth-capping redistribution pass.
//! * Canonical-Huffman code-length tree emission, reusing the 19-symbol
//!   meta-alphabet + run-length codes 16/17/18 expected by the decoder.
//! * A 4 KB sliding-window, hash-chain LZ77 matcher over the ARGB pixel
//!   sequence. Matches of length ≥ 3 are emitted as (length, distance)
//!   pairs using the VP8L length-or-distance symbol scheme. Distances are
//!   always emitted in the `code = d + 120` form, so the short-distance
//!   diamond table isn't consulted on the encoder side.
//!
//! The entry point is [`encode_vp8l_argb`]: a bare VP8L bitstream (no
//! RIFF wrapper) sized for a given `width × height` ARGB pixel buffer.

use oxideav_core::{Error, Result};

use super::VP8L_SIGNATURE;

/// Maximum Huffman code length allowed by the VP8L spec.
const MAX_CODE_LENGTH: u8 = 15;

/// LZ77 window size (in pixels). 4K pixels is plenty for the small-image
/// roundtrip tests and keeps the hash chain tight.
const LZ_WINDOW: usize = 4096;

/// Minimum LZ77 match length we're willing to emit. Shorter matches lose
/// to simple literals once the length/distance bits are counted.
const MIN_MATCH: usize = 3;

/// Maximum LZ77 match length. The VP8L length alphabet tops out well
/// above this but long runs are rare in ARGB data and short-chain hash
/// searches get expensive past a few hundred pixels.
const MAX_MATCH: usize = 4096;

/// LSB-first bit writer matching the VP8L decoder's bit-reader convention.
struct BitWriter {
    out: Vec<u8>,
    cur: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            cur: 0,
            nbits: 0,
        }
    }

    fn write(&mut self, value: u32, n: u32) {
        debug_assert!(n <= 32);
        let mask = if n == 0 {
            0u64
        } else if n == 32 {
            0xFFFF_FFFFu64
        } else {
            (1u64 << n) - 1
        };
        self.cur |= ((value as u64) & mask) << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push((self.cur & 0xff) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.cur & 0xff) as u8);
        }
        self.out
    }
}

/// Encode `width × height` ARGB pixels (one u32 per pixel: `a<<24 | r<<16 |
/// g<<8 | b`) as a bare VP8L bitstream (no RIFF wrapper).
///
/// `has_alpha` sets the `alpha_is_used` header bit. It's purely advisory
/// — the alpha channel is transmitted either way.
pub fn encode_vp8l_argb(
    width: u32,
    height: u32,
    pixels: &[u32],
    has_alpha: bool,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(Error::invalid("VP8L encoder: zero-size image"));
    }
    if width > 16384 || height > 16384 {
        return Err(Error::invalid("VP8L encoder: max dimension 16384"));
    }
    if (pixels.len() as u64) != (width as u64) * (height as u64) {
        return Err(Error::invalid("VP8L encoder: pixel count mismatch"));
    }

    let mut bw = BitWriter::new();
    // Signature.
    bw.write(VP8L_SIGNATURE as u32, 8);
    // 14-bit width-minus-1 / 14-bit height-minus-1. Decoder masks ×0x3fff
    // and reconstructs via `+1`, so cap at 16384 (done above).
    bw.write(width - 1, 14);
    bw.write(height - 1, 14);
    bw.write(if has_alpha { 1 } else { 0 }, 1);
    bw.write(0, 3); // version

    // No transforms (scoped non-goal for v1).
    bw.write(0, 1);

    // ── Main image stream ─────────────────────────────────────────────
    // No colour cache, no meta-Huffman image.
    bw.write(0, 1); // no colour cache
    bw.write(0, 1); // no meta-Huffman image (single group)

    // Step 1: build the LZ77-parsed symbol stream over the pixel array.
    let stream = build_symbol_stream(pixels);

    // Step 2: tally histograms for the five Huffman trees.
    let mut green_freq = vec![0u32; 256 + 24]; // 256 literals + 24 length codes
    let mut red_freq = vec![0u32; 256];
    let mut blue_freq = vec![0u32; 256];
    let mut alpha_freq = vec![0u32; 256];
    let mut dist_freq = vec![0u32; 40];

    for sym in &stream {
        match *sym {
            StreamSym::Literal { a, r, g, b } => {
                green_freq[g as usize] += 1;
                red_freq[r as usize] += 1;
                blue_freq[b as usize] += 1;
                alpha_freq[a as usize] += 1;
            }
            StreamSym::Backref {
                len_sym,
                dist_sym,
                ..
            } => {
                green_freq[256 + len_sym as usize] += 1;
                dist_freq[dist_sym as usize] += 1;
            }
        }
    }

    // Step 3: build canonical Huffman code lengths for each tree. The
    // decoder requires at least one non-zero length per tree; if the
    // stream never used (e.g.) red, we still have to emit a valid tree.
    let green_lens = build_limited_lengths(&green_freq, MAX_CODE_LENGTH)?;
    let red_lens = build_limited_lengths(&red_freq, MAX_CODE_LENGTH)?;
    let blue_lens = build_limited_lengths(&blue_freq, MAX_CODE_LENGTH)?;
    let alpha_lens = build_limited_lengths(&alpha_freq, MAX_CODE_LENGTH)?;
    let dist_lens = build_limited_lengths(&dist_freq, MAX_CODE_LENGTH)?;

    let green_codes = canonical_codes(&green_lens);
    let red_codes = canonical_codes(&red_lens);
    let blue_codes = canonical_codes(&blue_lens);
    let alpha_codes = canonical_codes(&alpha_lens);
    let dist_codes = canonical_codes(&dist_lens);

    // Step 4: emit the Huffman trees in the order the decoder expects.
    emit_huffman_tree(&mut bw, &green_lens)?;
    emit_huffman_tree(&mut bw, &red_lens)?;
    emit_huffman_tree(&mut bw, &blue_lens)?;
    emit_huffman_tree(&mut bw, &alpha_lens)?;
    emit_huffman_tree(&mut bw, &dist_lens)?;

    // Step 5: emit the symbol stream.
    for sym in &stream {
        match *sym {
            StreamSym::Literal { a, r, g, b } => {
                write_code(&mut bw, &green_codes, &green_lens, g as usize);
                write_code(&mut bw, &red_codes, &red_lens, r as usize);
                write_code(&mut bw, &blue_codes, &blue_lens, b as usize);
                write_code(&mut bw, &alpha_codes, &alpha_lens, a as usize);
            }
            StreamSym::Backref {
                len_sym,
                len_extra_bits,
                len_extra,
                dist_sym,
                dist_extra_bits,
                dist_extra,
            } => {
                write_code(&mut bw, &green_codes, &green_lens, 256 + len_sym as usize);
                if len_extra_bits > 0 {
                    bw.write(len_extra, len_extra_bits);
                }
                write_code(&mut bw, &dist_codes, &dist_lens, dist_sym as usize);
                if dist_extra_bits > 0 {
                    bw.write(dist_extra, dist_extra_bits);
                }
            }
        }
    }

    Ok(bw.finish())
}

/// Parsed-pixel symbol. Either a literal ARGB quadruplet or an LZ77
/// backreference (length + distance with their extra-bit fields already
/// factored out).
#[derive(Clone, Copy)]
enum StreamSym {
    Literal {
        a: u8,
        r: u8,
        g: u8,
        b: u8,
    },
    Backref {
        len_sym: u32,
        len_extra_bits: u32,
        len_extra: u32,
        dist_sym: u32,
        dist_extra_bits: u32,
        dist_extra: u32,
    },
}

/// Factor a VP8L length/distance *value* (≥1) into a prefix symbol +
/// trailing extra bits. Inverse of `decode_length_or_distance` in
/// `super`.
fn encode_len_or_dist_value(value: u32) -> (u32, u32, u32) {
    debug_assert!(value >= 1);
    if value <= 4 {
        return (value - 1, 0, 0);
    }
    let v = value - 1; // ≥ 4
    let msb = 31 - v.leading_zeros(); // ≥ 2
    let extra_bits = msb - 1;
    let sym_sub = (v >> extra_bits) & 1; // 0 (even) or 1 (odd)
    let symbol = 2 * extra_bits + 2 + sym_sub;
    let offset = (2 + sym_sub) << extra_bits;
    let extra = v - offset;
    (symbol, extra_bits, extra)
}

/// Walk `pixels` and emit literals + LZ77 backreferences. Uses a simple
/// prefix-hash chain with head + next-pointer arrays; the chain is
/// bounded by [`LZ_WINDOW`].
fn build_symbol_stream(pixels: &[u32]) -> Vec<StreamSym> {
    let mut out: Vec<StreamSym> = Vec::with_capacity(pixels.len());
    let n = pixels.len();
    // Hash table: 12-bit table, heads index into `pixels`. `next` is a
    // per-pixel chain pointer (usize::MAX = terminator).
    const HASH_BITS: u32 = 12;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    let mut head: Vec<usize> = vec![usize::MAX; HASH_SIZE];
    let mut next: Vec<usize> = vec![usize::MAX; n];

    let hash3 = |p0: u32, p1: u32, p2: u32| -> usize {
        // Cheap multiplicative hash over 3 pixels (12 bytes worth of data).
        let k = p0
            .wrapping_mul(0x9E3779B9)
            .wrapping_add(p1.wrapping_mul(0x85EBCA77))
            .wrapping_add(p2.wrapping_mul(0xC2B2AE3D));
        (k >> (32 - HASH_BITS)) as usize
    };

    let mut i = 0usize;
    while i < n {
        // Find best match starting at i, if at least MIN_MATCH pixels
        // remain.
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + MIN_MATCH <= n {
            let h = hash3(pixels[i], pixels[i + 1], pixels[i + 2]);
            // Walk the chain.
            let mut candidate = head[h];
            let mut tries = 64usize; // chain length cap
            while candidate != usize::MAX && tries > 0 {
                let dist = i - candidate;
                if dist == 0 || dist > LZ_WINDOW {
                    break;
                }
                // Measure the match length.
                let max_len = (n - i).min(MAX_MATCH);
                let mut l = 0usize;
                while l < max_len && pixels[candidate + l] == pixels[i + l] {
                    l += 1;
                }
                if l >= MIN_MATCH && l > best_len {
                    best_len = l;
                    best_dist = dist;
                    // Good enough heuristic: stop at 64 pixels.
                    if l >= 64 {
                        break;
                    }
                }
                candidate = next[candidate];
                tries -= 1;
            }
        }

        if best_len >= MIN_MATCH {
            // Emit a backreference. Length value = best_len, distance
            // value = best_dist. Distances use `code = d + 120` form so
            // we never touch the short-distance plane table.
            let (len_sym, len_eb, len_ex) = encode_len_or_dist_value(best_len as u32);
            let (dist_sym, dist_eb, dist_ex) =
                encode_len_or_dist_value((best_dist as u32) + 120);
            out.push(StreamSym::Backref {
                len_sym,
                len_extra_bits: len_eb,
                len_extra: len_ex,
                dist_sym,
                dist_extra_bits: dist_eb,
                dist_extra: dist_ex,
            });
            // Insert every covered pixel into the hash (but stop inserts
            // a hair before the end so we don't touch past the buffer).
            for k in 0..best_len {
                let pos = i + k;
                if pos + 2 < n {
                    let h = hash3(pixels[pos], pixels[pos + 1], pixels[pos + 2]);
                    next[pos] = head[h];
                    head[h] = pos;
                }
            }
            i += best_len;
        } else {
            // Literal: emit as ARGB quadruplet in green/red/blue/alpha
            // order (matching the decoder's per-channel tree layout).
            let p = pixels[i];
            out.push(StreamSym::Literal {
                a: ((p >> 24) & 0xff) as u8,
                r: ((p >> 16) & 0xff) as u8,
                g: ((p >> 8) & 0xff) as u8,
                b: (p & 0xff) as u8,
            });
            if i + 2 < n {
                let h = hash3(pixels[i], pixels[i + 1], pixels[i + 2]);
                next[i] = head[h];
                head[h] = i;
            }
            i += 1;
        }
    }
    out
}

/// Canonical Huffman code length builder with a 15-bit length cap.
///
/// Algorithm:
/// 1. Build a standard heap-based Huffman tree from `freqs`.
/// 2. If the deepest symbol exceeds `max_len`, redistribute code lengths
///    by borrowing from shorter codes (the classic "length-limited
///    Huffman" trick by Kraft-sum repair).
/// 3. Always return a valid tree — if only 0 or 1 symbols are live, we
///    add a synthetic second symbol so the decoder builds a branching
///    tree (VP8L's simple-code path is also valid but the decoder in
///    this crate handles both).
fn build_limited_lengths(freqs: &[u32], max_len: u8) -> Result<Vec<u8>> {
    let n = freqs.len();
    let mut lens = vec![0u8; n];
    let nonzero: Vec<usize> = (0..n).filter(|&i| freqs[i] > 0).collect();

    if nonzero.is_empty() {
        // No symbols used — give symbol 0 a length-of-1 code to keep the
        // tree well-formed. Also give symbol 1 (or `n-1` if n==1) so the
        // decoder's canonical builder sees two leaves.
        if n >= 2 {
            lens[0] = 1;
            lens[1] = 1;
        } else {
            lens[0] = 1;
        }
        return Ok(lens);
    }
    if nonzero.len() == 1 {
        // A single real symbol — pair it with a dummy to make the tree
        // non-degenerate. Both at length 1.
        let s = nonzero[0];
        lens[s] = 1;
        // Pick a different symbol for the dummy.
        let d = if s == 0 { 1.min(n - 1) } else { 0 };
        lens[d] = 1;
        return Ok(lens);
    }

    // Build the tree using a simple priority queue on (freq, node_id).
    // Nodes: leaves 0..n, internal nodes n..
    #[derive(Clone)]
    struct Node {
        freq: u64,
        left: i32,
        right: i32,
        symbol: i32, // leaves only
    }
    let mut nodes: Vec<Node> = Vec::with_capacity(n * 2);
    for (i, &f) in freqs.iter().enumerate() {
        nodes.push(Node {
            freq: f as u64,
            left: -1,
            right: -1,
            symbol: i as i32,
        });
    }
    // Min-heap of live node indices keyed by freq.
    let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> =
        std::collections::BinaryHeap::new();
    for &i in &nonzero {
        heap.push(std::cmp::Reverse((nodes[i].freq, i)));
    }
    while heap.len() > 1 {
        let std::cmp::Reverse((fa, a)) = heap.pop().unwrap();
        let std::cmp::Reverse((fb, b)) = heap.pop().unwrap();
        let idx = nodes.len();
        nodes.push(Node {
            freq: fa + fb,
            left: a as i32,
            right: b as i32,
            symbol: -1,
        });
        heap.push(std::cmp::Reverse((fa + fb, idx)));
    }
    let root = heap.pop().unwrap().0 .1;

    // Assign lengths via DFS.
    fn walk(nodes: &[Node], idx: usize, depth: u8, lens: &mut [u8]) {
        let n = &nodes[idx];
        if n.symbol >= 0 {
            lens[n.symbol as usize] = depth.max(1);
        } else {
            walk(nodes, n.left as usize, depth + 1, lens);
            walk(nodes, n.right as usize, depth + 1, lens);
        }
    }
    walk(&nodes, root, 0, &mut lens);

    // Length-limit the lengths.
    limit_code_lengths(&mut lens, max_len);
    Ok(lens)
}

/// Cap every non-zero code length at `max_len` by redistributing "Kraft
/// budget" from the deepest leaves up into shallower slots. This is the
/// classic post-hoc fix that works well enough for small alphabets.
fn limit_code_lengths(lens: &mut [u8], max_len: u8) {
    // Count symbols per length.
    let max_observed = *lens.iter().max().unwrap_or(&0);
    if max_observed <= max_len {
        return;
    }
    let mut bl_count: Vec<u32> = vec![0; (max_observed as usize + 1).max(1)];
    for &l in lens.iter() {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    // Move anything past max_len down to max_len — overflow causes
    // Kraft > 1, which we then repair by promoting lower-depth leaves.
    let mut overflow: u64 = 0;
    for l in (max_len as usize + 1)..bl_count.len() {
        // Each code at depth l counts for 2^(max_len - l) of Kraft
        // budget when collapsed to max_len (which is bigger than its
        // original slot since l > max_len → shift is negative → treat
        // as scaling up). We use a fixed-point budget of 2^max_len for
        // the full codespace.
        overflow += (bl_count[l] as u64) * ((1u64 << (l - max_len as usize)) - 1);
        bl_count[max_len as usize] += bl_count[l];
        bl_count[l] = 0;
    }

    // Now redistribute by borrowing from lengths < max_len.
    // overflow is expressed as multiples of 2^0 at depth max_len (i.e.,
    // each overflow unit = one extra leaf at max_len beyond what fits).
    // Repairing: take a leaf from depth d (d < max_len), move it to
    // depth max_len. That frees 2^(max_len - d) slots at max_len minus
    // 1 for the new leaf there = net gain of 2^(max_len - d) - 1 slots.
    while overflow > 0 {
        let mut d = max_len as i32 - 1;
        while d > 0 && bl_count[d as usize] == 0 {
            d -= 1;
        }
        if d <= 0 {
            break;
        }
        bl_count[d as usize] -= 1;
        bl_count[(d + 1) as usize] += 1;
        // Net change: split a leaf (counts for 2^(max_len - d) Kraft
        // units) into a leaf at d+1 (counts for 2^(max_len - d - 1))
        // + a free slot at d+1 (also 2^(max_len - d - 1)). The free
        // slot absorbs 1 unit of overflow.
        //
        // Overflow is tracked in "units of the max_len slot", so the
        // free slot at d+1 absorbs 2^(max_len - d - 1) units.
        let freed = 1u64 << ((max_len as i32 - d - 1).max(0) as u32);
        if freed >= overflow {
            overflow = 0;
        } else {
            overflow -= freed;
        }
    }

    // Rewrite lens by assigning new per-length counts, preserving
    // original symbol→length mapping as much as possible. We walk
    // symbols sorted by original depth and reassign them to the new
    // bl_count distribution.
    let mut by_depth: Vec<(u8, usize)> = lens
        .iter()
        .enumerate()
        .filter(|(_, &l)| l > 0)
        .map(|(i, &l)| (l, i))
        .collect();
    // Sort deepest-first so leaves that used to be deepest land at the
    // longest code lengths in the new distribution too.
    by_depth.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for l in lens.iter_mut() {
        *l = 0;
    }
    let mut idx = 0usize;
    // Fill from longest to shortest.
    for l in (1..=max_len as usize).rev() {
        let cnt = bl_count[l] as usize;
        for _ in 0..cnt {
            if idx >= by_depth.len() {
                break;
            }
            let (_, sym) = by_depth[idx];
            lens[sym] = l as u8;
            idx += 1;
        }
    }
    // Any leftover symbols (shouldn't happen if bl_count is consistent
    // with nonzero count) get the max length.
    while idx < by_depth.len() {
        let (_, sym) = by_depth[idx];
        lens[sym] = max_len;
        idx += 1;
    }
}

/// Assign canonical codes from code lengths, MSB-first within each code.
fn canonical_codes(lens: &[u8]) -> Vec<u32> {
    let max_len = *lens.iter().max().unwrap_or(&0);
    let mut codes = vec![0u32; lens.len()];
    if max_len == 0 {
        return codes;
    }
    let mut bl_count = vec![0u32; max_len as usize + 1];
    for &l in lens {
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    let mut next_code = vec![0u32; max_len as usize + 1];
    let mut code = 0u32;
    for bits in 1..=max_len as usize {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for (sym, &l) in lens.iter().enumerate() {
        if l > 0 {
            codes[sym] = next_code[l as usize];
            next_code[l as usize] += 1;
        }
    }
    codes
}

/// Write a canonical code for symbol `sym`, MSB first (matches the
/// decoder's walk through internal nodes). `codes[sym]` is the code value
/// with the most-significant bit at bit position `lens[sym]-1`.
fn write_code(bw: &mut BitWriter, codes: &[u32], lens: &[u8], sym: usize) {
    let l = lens[sym];
    let code = codes[sym];
    // Emit MSB first. VP8L packs LSB-first, so we bit-reverse the code
    // into a left-to-right LSB-first sequence.
    let mut rev = 0u32;
    for i in 0..l {
        if (code >> i) & 1 != 0 {
            rev |= 1 << (l - 1 - i);
        }
    }
    bw.write(rev, l as u32);
}

/// Emit a Huffman tree in VP8L "normal" form: fixed CODE_LENGTH_ORDER
/// header + per-symbol lengths via the meta-tree. Run-length codes
/// 16/17/18 are used for consecutive zeros and repeats.
fn emit_huffman_tree(bw: &mut BitWriter, lens: &[u8]) -> Result<()> {
    // simple-code = 0 (we always use the normal form).
    bw.write(0, 1);

    // Build the "code-length code lengths" — lengths for the 19-symbol
    // meta-alphabet used to encode `lens`.
    const CODE_LENGTH_ORDER: [usize; 19] = [
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];

    // Compress `lens` into a sequence of meta-symbols (0..15 for direct
    // lengths, 16 = copy previous length 3-6×, 17 = zero-run 3-10, 18 =
    // zero-run 11-138). For v1 we keep the RLE simple and only collapse
    // long zero runs (the common case) into code 17/18. Repeat-previous
    // (code 16) is left out — a handful of redundant bits on repeated
    // small-length plateaus is fine for a first cut.
    let meta_stream = compress_lengths(lens);

    // Tally meta-alphabet frequencies and build a (≤7-bit) tree for it.
    // The code-length alphabet limit is 7 bits per spec.
    let mut meta_freq = vec![0u32; 19];
    for (code, _extra) in &meta_stream {
        meta_freq[*code as usize] += 1;
    }
    let meta_lens = build_limited_lengths(&meta_freq, 7)?;
    let meta_codes = canonical_codes(&meta_lens);

    // Write the meta-alphabet lengths. num_code_lengths must cover
    // every *used* slot per CODE_LENGTH_ORDER, but we keep it simple and
    // always emit all 19 (num_code_lengths - 4 = 15, 4 bits).
    let mut last_used = 0usize;
    for i in 0..19 {
        let sym = CODE_LENGTH_ORDER[i];
        if meta_lens[sym] != 0 {
            last_used = i + 1;
        }
    }
    let num_code_lengths = last_used.max(4);
    bw.write((num_code_lengths - 4) as u32, 4);
    for i in 0..num_code_lengths {
        let sym = CODE_LENGTH_ORDER[i];
        bw.write(meta_lens[sym] as u32, 3);
    }

    // No truncation mode (bit = 0 → decode the entire `lens` alphabet).
    bw.write(0, 1);

    // Emit the meta stream.
    for (code, extra) in &meta_stream {
        write_code(bw, &meta_codes, &meta_lens, *code as usize);
        match *code {
            16 => bw.write(*extra, 2), // 3 + 2-bit extra
            17 => bw.write(*extra, 3), // 3 + 3-bit extra
            18 => bw.write(*extra, 7), // 11 + 7-bit extra
            _ => {}
        }
    }
    Ok(())
}

/// RLE-compress code lengths into a (meta_symbol, extra_bits) stream.
/// Zero runs collapse into codes 17 / 18; non-zero values emit code 0..15
/// verbatim. Code 16 (repeat-previous-length) is currently unused for
/// simplicity.
fn compress_lengths(lens: &[u8]) -> Vec<(u8, u32)> {
    let mut out: Vec<(u8, u32)> = Vec::new();
    let mut i = 0usize;
    while i < lens.len() {
        let v = lens[i];
        if v == 0 {
            // Count zero run.
            let mut j = i;
            while j < lens.len() && lens[j] == 0 {
                j += 1;
            }
            let mut run = j - i;
            // Fold with code 17 (run 3..10) and 18 (run 11..138).
            while run >= 11 {
                let take = run.min(138);
                out.push((18, (take - 11) as u32));
                run -= take;
            }
            while run >= 3 {
                let take = run.min(10);
                out.push((17, (take - 3) as u32));
                run -= take;
            }
            for _ in 0..run {
                out.push((0, 0));
            }
            i = j;
        } else {
            out.push((v, 0));
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_len_dist_roundtrip_small() {
        for value in 1u32..=200 {
            let (sym, eb, extra) = encode_len_or_dist_value(value);
            // Re-derive via the decoder's formula.
            let decoded = if sym < 4 {
                sym + 1
            } else {
                let eb_d = (sym - 2) >> 1;
                let off = (2 + (sym & 1)) << eb_d;
                off + extra + 1
            };
            assert_eq!(decoded, value, "round-trip failed for value {value}");
            assert!(eb <= 14, "extra_bits too big for value {value}");
            let _ = extra;
            let _ = eb;
        }
    }

    #[test]
    fn canonical_codes_match_decoder_shape() {
        let lens = [2u8, 1u8, 3u8, 3u8];
        let codes = canonical_codes(&lens);
        // Canonical assignment: shortest first. Expected (MSB-first):
        //   sym1 (len 1): 0
        //   sym0 (len 2): 10
        //   sym2 (len 3): 110
        //   sym3 (len 3): 111
        assert_eq!(codes[1], 0b0);
        assert_eq!(codes[0], 0b10);
        assert_eq!(codes[2], 0b110);
        assert_eq!(codes[3], 0b111);
    }
}
