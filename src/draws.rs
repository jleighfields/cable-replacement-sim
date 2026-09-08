//! Random draws computed from their index rather than from a stream position.
//!
//! Every uniform this simulation needs is a pure function of where it sits:
//!
//! ```text
//! draw(purpose, replication, segment, year)
//! ```
//!
//! Nothing is stored, nothing advances, and asking for one draw does not change
//! what any other draw will be. Three properties follow, and each is load-bearing
//! somewhere in this project:
//!
//! * **Two policies see the same numbers.** A comparison between policies is a
//!   paired difference only if replication `r` meets identical lifetimes under
//!   each. With a stream that would require both policies to consume it in the
//!   same order, which they do not — a policy that replaces more segments reads
//!   more draws. Indexing by position removes the question.
//! * **Workers share nothing.** A stream has state, so several threads drawing
//!   from one need a lock, and the order they take their turns in decides the
//!   answer. Here a worker computes what it needs and coordinates with nobody,
//!   so the result does not depend on the thread count or on scheduling.
//! * **Only what is read is computed.** A year's draw is consumed only by a
//!   segment replaced in that year, which is a few percent of them. A stream
//!   would have to produce the rest anyway to keep its position aligned.
//!
//! # The generator
//!
//! Philox-4x64-10, from the Random123 family. It is a *counter-based* generator:
//! rather than evolving a state, it encrypts a counter under a key, so the value
//! at any index is available without producing the ones before it.
//!
//! **NumPy ships the same generator, and that is why this one is here.** The
//! Python reference and the batched implementations still take their draws as
//! arrays, so both sides have to produce the same numbers. Agreeing with a
//! published algorithm that each implements separately is a stronger position
//! than agreeing with each other — and the test compares this against NumPy's
//! output directly.
//!
//! # Reading this beside the Python
//!
//! * **`u64` arithmetic wraps only where it is asked to.** Rust panics on
//!   overflow in a debug build rather than wrapping silently, so the key
//!   schedule says `wrapping_add`. Python's integers grow instead, so NumPy's
//!   equivalent is written with an explicit 64-bit type.
//! * **`u128` is how the 64x64 product is taken.** Multiplying two `u64` values
//!   gives 128 bits, and both halves are needed. Python would take the product
//!   in arbitrary precision and shift.

/// First multiplier of the Philox-4x64 round function.
const MULTIPLIER_0: u64 = 0xD2E7_470E_E14C_6C93;

/// Second multiplier of the Philox-4x64 round function.
const MULTIPLIER_1: u64 = 0xCA5A_8263_9512_1157;

/// Key increment per round, from the golden ratio.
const WEYL_0: u64 = 0x9E37_79B9_7F4A_7C15;

/// Key increment per round, from the square root of three.
const WEYL_1: u64 = 0xBB67_AE85_84CA_A73B;

/// Rounds in Philox-4x64-10, which is the variant NumPy implements.
const ROUNDS: usize = 10;

/// How many 64-bit values one counter yields.
///
/// The generator produces four at a time, so a position in the flat stream of
/// draws is a block and a lane within it: index `i` comes from block `i / 4`,
/// lane `i % 4`.
pub const LANES: usize = 4;

/// What NumPy's counter is ahead of the block index by.
///
/// **NumPy increments the counter before producing each block**, so the first
/// block it emits for a given counter is the one Philox defines at that counter
/// plus one. Both sides have to agree on this or every draw is one block out —
/// which looks like perfectly good randomness and is wrong everywhere. It is
/// checked against NumPy rather than reasoned about, because reasoning about it
/// is what produced the wrong answer first.
const NUMPY_COUNTER_LEAD: u64 = 1;

/// The high and low halves of a 64-by-64 bit product.
///
/// # Arguments
///
/// * `left` - one factor.
/// * `right` - the other factor.
fn multiply_wide(left: u64, right: u64) -> (u64, u64) {
    let product = u128::from(left) * u128::from(right);
    ((product >> 64) as u64, product as u64)
}

/// One Philox-4x64 round.
///
/// # Arguments
///
/// * `counter` - the four-word counter being encrypted.
/// * `key` - the two-word key for this round.
fn round(counter: [u64; LANES], key: [u64; 2]) -> [u64; LANES] {
    let (high_0, low_0) = multiply_wide(MULTIPLIER_0, counter[0]);
    let (high_1, low_1) = multiply_wide(MULTIPLIER_1, counter[2]);
    [
        high_1 ^ counter[1] ^ key[0],
        low_1,
        high_0 ^ counter[3] ^ key[1],
        low_0,
    ]
}

/// Encrypts one counter under one key, giving four uniform 64-bit words.
///
/// This is the whole generator. There is no state to carry between calls, which
/// is what makes it safe to call from any thread at any time.
///
/// # Arguments
///
/// * `counter` - the position being drawn for.
/// * `key` - derived from the run's seed.
///
/// # Returns
///
/// Four uniformly distributed 64-bit words.
pub fn philox(counter: [u64; LANES], key: [u64; 2]) -> [u64; LANES] {
    let mut state = counter;
    let mut schedule = key;
    for index in 0..ROUNDS {
        if index > 0 {
            schedule = [
                schedule[0].wrapping_add(WEYL_0),
                schedule[1].wrapping_add(WEYL_1),
            ];
        }
        state = round(state, schedule);
    }
    state
}

/// Converts one 64-bit word to a double in `[0, 1)`.
///
/// The top 53 bits are kept, which is every bit a double can represent without
/// rounding, and the result is scaled by two to the fifty-third. **This is
/// NumPy's conversion**, and it has to be exactly NumPy's: the Python side
/// produces the same uniforms for the reference implementation, and a different
/// rounding here would make the two disagree in the last bit of every draw.
///
/// # Arguments
///
/// * `word` - a uniform 64-bit word.
pub fn to_double(word: u64) -> f64 {
    (word >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// The four words at one block of the stream, under NumPy's counter convention.
///
/// This is the function everything else should call: it takes the block index a
/// caller reasons about and applies the off-by-one that NumPy's pre-increment
/// introduces, so no other code has to know about it.
///
/// # Arguments
///
/// * `block` - which group of four words to produce.
/// * `key` - derived from the run's seed.
///
/// # Returns
///
/// The same four words `numpy.random.Philox(key, counter=block).random_raw(4)`
/// returns.
pub fn block(block: u64, key: [u64; 2]) -> [u64; LANES] {
    philox([block + NUMPY_COUNTER_LEAD, 0, 0, 0], key)
}

/// Which stream a draw belongs to, kept apart by a field of its index.
///
/// The separation is load-bearing rather than tidy: without it the
/// replacement-policy priorities would come out correlated with the same
/// segments' lifetime draws, and since a larger uniform gives a shorter
/// lifetime the random policy would rank segments by imminence of failure and
/// stop being a control. The names are read on the Python side, so they are
/// authored here and nowhere else.
pub mod purpose {
    /// Segment lifetime draws.
    pub const LIFETIMES: u64 = 0;
    /// Replacement-policy randomness.
    pub const POLICIES: u64 = 1;
    /// The synthetic segment table.
    pub const POPULATION: u64 = 2;
    /// The synthetic censored failure history.
    pub const RECORDS: u64 = 3;
}

/// Bits reserved for the year within a draw's index.
///
/// Sixty-four years of horizon. A simulation is thirty, and a study that wanted
/// more would be a different question than this one answers.
const YEAR_BITS: u64 = 6;

/// Bits reserved for the segment.
const SEGMENT_BITS: u64 = 32;

/// Bits reserved for the replication.
const REPLICATION_BITS: u64 = 20;

/// Where the segment field sits: the low bits, deliberately.
///
/// **Consecutive segments have to land on consecutive indices**, because the
/// generator produces four words at a time and one block is therefore shared by
/// four adjacent indices. With the segment field anywhere else, segment `s` and
/// segment `s + 1` would be at least sixty-four apart, so every draw would cost
/// a full encryption of which three quarters was thrown away. Here four
/// adjacent segments come out of one call, which is what makes the dense case
/// affordable — every segment's starting lifetime, and every segment's
/// priority.
pub const SEGMENT_SHIFT: u64 = 0;
/// Where the year field starts, above the segment.
pub const YEAR_SHIFT: u64 = SEGMENT_BITS;
/// Where the replication field starts.
pub const REPLICATION_SHIFT: u64 = YEAR_SHIFT + YEAR_BITS;
/// Where the purpose field starts, leaving six bits above it.
pub const PURPOSE_SHIFT: u64 = REPLICATION_SHIFT + REPLICATION_BITS;

/// Bits left for the purpose above the other three fields.
const PURPOSE_BITS: u64 = 64 - (REPLICATION_BITS + YEAR_BITS + SEGMENT_BITS);

/// The largest purpose an index can carry.
///
/// **Bounded like the other three, because it folds the same way.** It sits in
/// the top bits, so a purpose past this shifts out of the word entirely and
/// lands on one that fits — identically on both sides of the boundary, which is
/// why no comparison between implementations could see it. This field is what
/// keeps the replacement-policy priorities off the same segments' lifetime
/// draws, so a fold here is exactly the correlation the separation exists to
/// prevent.
pub const MAX_PURPOSE: u64 = (1 << PURPOSE_BITS) - 1;

/// The largest year an index can carry.
pub const MAX_YEAR: u64 = (1 << YEAR_BITS) - 1;
/// The largest segment identifier an index can carry.
pub const MAX_SEGMENT: u64 = (1 << SEGMENT_BITS) - 1;
/// The largest replication an index can carry.
pub const MAX_REPLICATION: u64 = (1 << REPLICATION_BITS) - 1;

// The shipped run has to fit with room to spare, and narrowing a field is the
// way that would quietly stop being true — two positions would then share one
// draw, which is a correlation nothing downstream could detect. Checked when
// this compiles rather than by a test, because a test comparing one constant
// against another can never fail and is worth nothing.
const _: () = assert!(12_000 < MAX_SEGMENT, "the shipped population must fit");
const _: () = assert!(1_000 < MAX_REPLICATION, "the shipped run must fit");
const _: () = assert!(30 < MAX_YEAR, "the shipped horizon must fit");
// And the fields must not overlap: together they have to leave room above.
const _: () = assert!(YEAR_BITS + SEGMENT_BITS + REPLICATION_BITS < 64);
// The purpose takes whatever is left, so every bit of the word is accounted
// for and no field can be widened without narrowing another on purpose.
const _: () = assert!(
    PURPOSE_BITS + REPLICATION_BITS + YEAR_BITS + SEGMENT_BITS == 64,
    "the four fields must tile the word"
);
// Every purpose this crate names has to fit the field it sits in. Checked when
// this compiles, which is why the annual loop's own draws need no run-time
// purpose check — only a caller supplying one can get it wrong.
//
// The whole set, rather than whichever constant is largest today: asserting one
// of them would keep compiling when a fifth purpose is added past the field.
// Adding a purpose means adding it here, which is the point.
const NAMED_PURPOSES: [u64; 4] = [
    purpose::LIFETIMES,
    purpose::POLICIES,
    purpose::POPULATION,
    purpose::RECORDS,
];
const _: () = {
    let mut index = 0;
    while index < NAMED_PURPOSES.len() {
        assert!(NAMED_PURPOSES[index] <= MAX_PURPOSE, "a purpose must fit");
        index += 1;
    }
};

/// Where in the stream the draw for one position of the simulation lives.
///
/// **Fixed bit fields rather than a product of the run's dimensions.** Packing
/// as `((purpose * n_reps + replication) * n_segments + segment) * years + year`
/// would work, but it would make a segment's draws depend on how many segments
/// the run happened to have — so resizing the population would reshuffle
/// everyone's randomness, and a small run would stop being a smaller version of
/// a large one. With fixed fields, segment 5 of replication 3 in year 2 draws
/// the same number whatever else is in the run.
///
/// The purpose field is what keeps the replacement-policy priorities from
/// correlating with the same segments' lifetime draws. That separation is
/// load-bearing rather than tidy: a larger uniform gives a shorter lifetime, so
/// without it the random policy would rank segments by imminence of failure and
/// stop being a control.
///
/// # Arguments
///
/// * `purpose` - which stream, keeping unrelated draws independent. At most
///   `MAX_PURPOSE`.
/// * `replication` - at most `MAX_REPLICATION`.
/// * `segment` - at most `MAX_SEGMENT`.
/// * `year` - at most `MAX_YEAR`.
///
/// # Returns
///
/// The position in the stream, which is unique for each distinct argument set.
pub fn index(purpose: u64, replication: u64, segment: u64, year: u64) -> u64 {
    // `assert!`, not `debug_assert!`. The shipped build is a release build, so
    // a debug assertion here would be compiled out of everything that runs and
    // the injective packing would rest entirely on the boundary guards. Three
    // comparisons against the cost of an encryption is not measurable.
    assert!(purpose <= MAX_PURPOSE, "purpose out of range");
    assert!(replication <= MAX_REPLICATION, "replication out of range");
    assert!(segment <= MAX_SEGMENT, "segment out of range");
    assert!(year <= MAX_YEAR, "year out of range");
    (purpose << PURPOSE_SHIFT)
        | (replication << REPLICATION_SHIFT)
        | (year << YEAR_SHIFT)
        | (segment << SEGMENT_SHIFT)
}

/// The uniform at one position in the flat stream of draws.
///
/// A position is a block and a lane within it, which is what lets any draw be
/// produced on its own: index `i` comes from block `i / 4`, lane `i % 4`.
///
/// # Arguments
///
/// * `index` - the position in the stream.
/// * `key` - derived from the run's seed.
///
/// # Returns
///
/// A double in `[0, 1)`, equal to what NumPy produces at the same position.
pub fn uniform_at(index: u64, key: [u64; 2]) -> f64 {
    let lanes = LANES as u64;
    to_double(block(index / lanes, key)[(index % lanes) as usize])
}

/// Fills a slice with the uniforms at a run of consecutive positions.
///
/// **One encryption per four draws, rather than one per draw.** The generator
/// produces four words at a time and the segment field sits in the low bits
/// precisely so that adjacent segments land on adjacent indices — but
/// `uniform_at` computes a whole block and returns a single lane, so a loop
/// calling it for consecutive segments encrypts each block four times and
/// discards three quarters of every one.
///
/// # Arguments
///
/// * `first_index` - the position the run starts at.
/// * `key` - the two key words for this run.
/// * `into` - filled with one uniform per position, in order.
pub fn fill_run<T: crate::weibull::Real>(first_index: u64, key: [u64; 2], into: &mut [T]) {
    let lanes = LANES as u64;
    let mut position = first_index;
    let mut filled = 0;
    while filled < into.len() {
        let words = block(position / lanes, key);
        // A run need not start on a block boundary, so the first block is
        // entered part-way through.
        let mut lane = (position % lanes) as usize;
        while lane < LANES && filled < into.len() {
            // Produced in double and narrowed as it is written. The
            // generator is checked against NumPy's own Philox, so it computes
            // at one width whatever the run's precision is, and the narrowing
            // is the same rounding NumPy applies on the other side.
            into[filled] = T::from_double(to_double(words[lane]));
            filled += 1;
            lane += 1;
            position += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first eight raw words NumPy produces for `key=[1, 0]`, counter zero.
    ///
    /// Taken from `numpy.random.Philox(key=[1, 0], counter=[0, 0, 0, 0])` and
    /// pasted here, so this test compares against NumPy's implementation rather
    /// than against another copy of the same reasoning. The second four are the
    /// point of the pair: they show that counter 1 continues where counter 0
    /// stopped, which is the mapping from a flat stream index to a counter.
    const NUMPY_KEY_ONE: [u64; 8] = [
        0x4db6_a27b_7562_82df,
        0xd944_fa03_babe_0e2f,
        0x27f8_72e5_7706_0d32,
        0x07f6_9769_6a04_82a2,
        0xe677_fe4b_bd04_52ec,
        0x0d54_3dba_56d1_e799,
        0xbebe_12ca_d0eb_4d9e,
        0x3f0b_4abd_55f6_1f3d,
    ];

    #[test]
    fn the_first_block_matches_numpy() {
        assert_eq!(block(0, [1, 0]), NUMPY_KEY_ONE[0..4]);
    }

    #[test]
    fn a_block_continues_where_the_one_before_it_stopped() {
        // What makes a flat index divisible into a block and a lane. Without
        // it, index 4 would have to come from somewhere this cannot address.
        assert_eq!(block(1, [1, 0]), NUMPY_KEY_ONE[4..8]);
    }

    #[test]
    fn the_raw_generator_sits_one_block_behind_numpy() {
        // Pins the off-by-one itself rather than only its consequences. Without
        // this, someone calling `philox` directly would get draws one block out
        // from the Python side, which looks like good randomness everywhere and
        // is wrong everywhere.
        assert_eq!(philox([1, 0, 0, 0], [1, 0]), NUMPY_KEY_ONE[0..4]);
        assert_ne!(philox([0, 0, 0, 0], [1, 0]), NUMPY_KEY_ONE[0..4]);
    }

    #[test]
    fn every_draw_falls_inside_the_unit_interval() {
        // The half-open range matters: `draw_lifetime` takes `ln(1 - u)`, which
        // is infinite at exactly 1.
        for index in 0..64u64 {
            for word in block(index, [7, 11]) {
                let draw = to_double(word);
                assert!((0.0..1.0).contains(&draw), "{draw} outside [0, 1)");
            }
        }
    }

    #[test]
    fn every_field_of_an_index_is_distinguishable() {
        // Each field moved on its own must land somewhere different. A shift
        // that overlapped its neighbour would alias two positions onto one
        // draw, which is a correlation nothing downstream could detect.
        let base = index(0, 0, 0, 0);
        for moved in [
            index(1, 0, 0, 0),
            index(0, 1, 0, 0),
            index(0, 0, 1, 0),
            index(0, 0, 0, 1),
        ] {
            assert_ne!(base, moved);
        }
        // And the fields must not run into each other at their limits.
        assert_ne!(index(0, 0, MAX_SEGMENT, 0), index(0, 0, 0, 1));
        assert_ne!(index(0, 0, 0, MAX_YEAR), index(0, 1, 0, 0));
        assert_ne!(index(0, MAX_REPLICATION, 0, 0), index(1, 0, 0, 0));
    }

    #[test]
    fn adjacent_segments_share_a_block() {
        // The reason the segment field is at the bottom. Four adjacent segments
        // have to come out of one encryption, or three quarters of the
        // generator's work is discarded on the dense path — which is every
        // segment's starting lifetime and every segment's priority.
        let first = index(0, 3, 0, 7);
        for segment in 0..LANES as u64 {
            assert_eq!(index(0, 3, segment, 7), first + segment);
        }
    }

    #[test]
    fn a_run_gives_what_asking_one_at_a_time_gives() {
        // The whole point of filling a run is that it encrypts a quarter as
        // often. It has to agree with the position-at-a-time path exactly, and
        // at every offset into the first block, since a run need not start on a
        // boundary.
        let key = [0xDEAD_BEEF, 0x0BAD_C0DE];
        for first in [0u64, 1, 2, 3, 4, 7, 4096, 4099] {
            for length in [0usize, 1, 3, 4, 5, 9, 33] {
                let mut run = vec![0.0f64; length];
                fill_run(first, key, &mut run);
                for (offset, drawn) in run.iter().enumerate() {
                    assert_eq!(*drawn, uniform_at(first + offset as u64, key));
                }

                // The single-precision instantiation is a different generated
                // function, and it is the one a run at that width calls. The
                // draw is produced in double and narrowed on the way into the
                // slice, so what it must equal is the narrowed double.
                //
                // Narrowed with `as` rather than through `Real::from_double`,
                // which is what `fill_run` uses: routing both sides through the
                // same conversion would move them together, and the arm could
                // then not fail for a defect in that conversion — which is the
                // one it is here to catch.
                let mut narrow = vec![0.0f32; length];
                fill_run(first, key, &mut narrow);
                for (offset, drawn) in narrow.iter().enumerate() {
                    assert_eq!(*drawn, uniform_at(first + offset as u64, key) as f32);
                }
            }
        }
    }

    #[test]
    fn a_changed_key_changes_the_stream() {
        // Guards the key actually reaching the round function. Dropping it
        // there leaves a generator that still looks random and gives every run
        // the same numbers.
        assert_ne!(block(0, [1, 0]), block(0, [2, 0]));
    }
}
