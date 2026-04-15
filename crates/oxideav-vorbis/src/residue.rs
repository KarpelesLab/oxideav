//! Vorbis residue decoding (types 0, 1, 2).
//!
//! Reference: Vorbis I §8 — the residue layer turns a sequence of codebook
//! entries into a per-channel spectral residual that gets multiplied with
//! the floor curve to form the final frequency-domain spectrum.
//!
//! Type 2 is the most common in practice and the only path tested
//! end-to-end here. Types 0 and 1 share the same classification structure
//! and use the deinterleaved `decode_partition` helper directly.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::codebook::Codebook;
use crate::setup::Residue;

/// Decode `n_channels` residue vectors of length `n` from the bitstream and
/// add them into `vectors[ch][bin]`. Channels marked `do_not_decode[ch] =
/// true` are skipped (still consume zero bits but produce no output).
pub fn decode_residue(
    residue: &Residue,
    codebooks: &[Codebook],
    n: usize,
    do_not_decode: &[bool],
    vectors: &mut [Vec<f32>],
    br: &mut BitReader<'_>,
) -> Result<()> {
    let n_channels = vectors.len();
    if do_not_decode.len() != n_channels {
        return Err(Error::invalid("residue: do_not_decode length mismatch"));
    }
    if residue.kind == 2 {
        // Type 2: decode a single interleaved vector of length n_channels * n,
        // then deinterleave into per-channel slots.
        let total_len = n_channels * n;
        // If every channel is "do not decode", the entire residue is zero.
        if do_not_decode.iter().all(|&v| v) {
            return Ok(());
        }
        let mut interleaved = vec![0f32; total_len];
        decode_partitioned(
            residue,
            codebooks,
            total_len,
            &mut [&mut interleaved[..]],
            br,
        )?;
        // Deinterleave: sample i*n_channels + ch goes into channel ch.
        for ch in 0..n_channels {
            if do_not_decode[ch] {
                continue;
            }
            for i in 0..n {
                vectors[ch][i] += interleaved[i * n_channels + ch];
            }
        }
        Ok(())
    } else {
        // Type 0 / 1: per-channel decode (same partition layout).
        let mut slots: Vec<&mut [f32]> = vectors.iter_mut().map(|v| v.as_mut_slice()).collect();
        decode_partitioned(residue, codebooks, n, &mut slots[..], br)?;
        // For type 0, the codebook entries are deinterleaved within each
        // partition (entry vector dim defines stride). Type 1 stores them
        // concatenated. Our `decode_partitioned` handles type 1 layout
        // (concatenated). Type 0 deinterleaving is a TODO; rare in practice.
        if residue.kind == 0 {
            return Err(Error::unsupported(
                "Vorbis residue type 0 deinterleave not implemented",
            ));
        }
        Ok(())
    }
}

/// Shared partition/cascade decode loop for residue types 0/1/2.
///
/// `vectors` carries one or more output channel slices, each of length
/// `n`. For type 2 the caller pre-flattens to a single slice; for types
/// 0/1 each entry is a per-channel slice.
fn decode_partitioned(
    residue: &Residue,
    codebooks: &[Codebook],
    n: usize,
    vectors: &mut [&mut [f32]],
    br: &mut BitReader<'_>,
) -> Result<()> {
    let psz = residue.partition_size as usize;
    let begin = residue.begin as usize;
    let end = (residue.end as usize).min(n);
    if end <= begin {
        return Ok(());
    }
    let n_to_read = end - begin;
    if n_to_read % psz != 0 {
        return Err(Error::invalid(
            "Vorbis residue: (end-begin) not divisible by partition_size",
        ));
    }
    let n_partitions = n_to_read / psz;
    let n_channels = vectors.len();

    let classbook = &codebooks[residue.classbook as usize];
    let classwords_per_codeword = classbook.dimensions as usize;
    let classifications = residue.classifications as usize;
    let trace = std::env::var_os("OXIDEAV_VORBIS_TRACE").is_some();
    if trace {
        eprintln!(
            "[vorbis] residue kind={} begin={} end={} psz={} parts={} classbook_dim={} classes={} bitpos={}",
            residue.kind,
            begin,
            end,
            psz,
            n_partitions,
            classwords_per_codeword,
            classifications,
            br.bit_position()
        );
    }

    // Pre-decode per-channel partition class assignments. For each pass
    // (cascade level), partitions whose class has a book at that pass get
    // their values read.
    // Number of classifications-assignments to decode = ceil(n_partitions / classwords_per_codeword).
    let n_class_codewords = n_partitions.div_ceil(classwords_per_codeword);
    let mut classifications_table: Vec<Vec<u32>> = vec![vec![0; n_partitions]; n_channels];

    // Vorbis cascades 8 passes; on pass 0 we also fetch the per-partition class.
    // Per Vorbis I §8.6.3, hitting end-of-packet mid-decode is NOT an error —
    // remaining partitions/passes stay zero-filled. We use `Error::Eof` from
    // the bit reader as the EOP signal.
    'cascade: for pass in 0..8u32 {
        let mut partition_idx = 0usize;
        while partition_idx < n_partitions {
            if pass == 0 {
                for ch in 0..n_channels {
                    let class_id = match classbook.decode_scalar(br) {
                        Ok(v) => v,
                        Err(Error::Eof) => break 'cascade,
                        Err(e) => return Err(e),
                    };
                    if trace {
                        eprintln!(
                            "[vorbis] part_idx={} ch={} class_id={} bitpos={}",
                            partition_idx,
                            ch,
                            class_id,
                            br.bit_position()
                        );
                    }
                    // Vorbis I §8.6.2 step 6: decompose into
                    // base-`classifications` digits HIGH-DIGIT-FIRST. The
                    // LAST partition in the group gets class_id % classifications;
                    // the FIRST partition gets the high-order digit.
                    let mut tmp = class_id;
                    for i in (0..classwords_per_codeword).rev() {
                        if partition_idx + i < n_partitions {
                            classifications_table[ch][partition_idx + i] =
                                tmp % classifications as u32;
                        }
                        tmp /= classifications as u32;
                    }
                }
            }
            // Decode `classwords_per_codeword` partitions per outer step.
            for k in 0..classwords_per_codeword {
                let pidx = partition_idx + k;
                if pidx >= n_partitions {
                    break;
                }
                for ch in 0..n_channels {
                    let class_id = classifications_table[ch][pidx] as usize;
                    let book_index = residue.books[class_id][pass as usize];
                    if book_index < 0 {
                        if trace {
                            eprintln!(
                                "[vorbis] pass={} pidx={} ch={} class={} no book — skip",
                                pass, pidx, ch, class_id
                            );
                        }
                        continue;
                    }
                    let book = &codebooks[book_index as usize];
                    let dim = book.dimensions as usize;
                    let bin_start = begin + pidx * psz;
                    let bin_end = bin_start + psz;
                    if trace {
                        eprintln!(
                            "[vorbis] pass={} pidx={} ch={} class={} book={} dim={} bitpos={}",
                            pass,
                            pidx,
                            ch,
                            class_id,
                            book_index,
                            dim,
                            br.bit_position()
                        );
                    }
                    let mut bin = bin_start;
                    while bin < bin_end {
                        let entry = match book.decode_scalar(br) {
                            Ok(v) => v,
                            Err(Error::Eof) => break 'cascade,
                            Err(e) => return Err(e),
                        };
                        let vq = book.vq_lookup(entry)?;
                        for j in 0..dim {
                            if bin + j < bin_end && bin + j < vectors[ch].len() {
                                vectors[ch][bin + j] += vq[j];
                            }
                        }
                        bin += dim;
                    }
                }
            }
            partition_idx += classwords_per_codeword;
            let _ = n_class_codewords;
        }
    }
    Ok(())
}
