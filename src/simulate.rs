//! The annual simulation loop, one replication at a time.
//!
//! The Rust side of the mirror: `python/cablesim/simulate.py` is the
//! correctness reference, written to be checkable by reading, and this module
//! is the same loop with the same three numbered steps in the same order. When
//! the two disagree, the reference arbitrates.
//!
//! Two conventions are load-bearing, and getting either wrong leaves a run that
//! completes with plausible-looking curves:
//!
//! **Failure times are simulation time measured from year 0, never ages.** A
//! segment starting at `age0` with a drawn age-at-failure `T` fails at
//! `T - age0`. The loop compares against year boundaries, so holding an age
//! instead would be wrong by `age0`, which ranges over decades here.
//!
//! **A replacement enters service at the start of the following year.** A
//! segment replaced in year `y` is age 0 when year `y + 1` is scored. The age
//! it carries for the rest of year `y` is never read: it cannot fail again,
//! because its next failure time is at least `y + 1`, and it cannot be planned
//! work, because it has already been replaced this year. That is what keeps the
//! year loop free of any inner iteration, and what makes the deterministic
//! parity test terminate when lifetimes are forced to zero.
//!
//! # Reading this beside the Python
//!
//! The reference is written in NumPy: it operates on whole arrays, so a year's
//! failures are a boolean mask and a year's totals come from `bincount`. This
//! loop is written one segment at a time, and that is the main thing to hold in
//! mind while reading the two side by side. Neither form is a translation of
//! the other; each is what its language is fast at, and the parity tests are
//! what establish they compute the same numbers.
//!
//! Four differences account for most of what looks unfamiliar:
//!
//! * **The result buffers are flat.** Python holds seven arrays of shape
//!   `(replications, years, classes)` and indexes them with three subscripts.
//!   Here each is one long `Vec<f64>` and the caller computes the offset —
//!   `(replication * n_years + year) * n_classes + class`. That is exactly
//!   what NumPy does underneath for a C-ordered array; the arithmetic is
//!   visible because Rust has no built-in multidimensional array type, and the
//!   binding hands the buffer back to NumPy to be reshaped without copying.
//! * **Working buffers are allocated once, before the replication loop.**
//!   Python writes `age = age0.astype(float, copy=True)` inside the loop and
//!   lets the garbage collector reclaim last year's array. Rust has no garbage
//!   collector — memory is freed when its owner goes out of scope — so
//!   allocating outside the loop and overwriting with `copy_from_slice` avoids
//!   asking the allocator for the same twelve thousand elements thirty times a
//!   replication. This is the one place the two files differ in structure for
//!   a reason that is about the language rather than about the model.
//! * **Numeric types never convert themselves.** `year` is a `usize`, an
//!   unsigned integer sized to the machine's pointer, and comparing it to a
//!   `f64` requires writing `year as f64`. Python promotes `int` to `float`
//!   silently; Rust refuses to, which is why casts appear at every boundary
//!   between an index and a quantity.
//! * **`&[f64]` in, `Vec<f64>` out.** The inputs are borrowed views the
//!   function may read and not keep; the results are owned lists it built and
//!   hands over. Python makes no such distinction, and it is worth reading the
//!   signature with it in mind: everything with an `&` belongs to the caller.

use rayon::prelude::*;

use crate::draws;
use crate::policies::{self, Resolved};
use crate::weibull;
use crate::weibull::Real;

/// The lifetime uniform one segment reads in one year of one replication.
///
/// A helper because the address is four fields and writing them out at each of
/// the three call sites would put the same arithmetic in three places.
///
/// # Arguments
///
/// * `key` - the two key words for this run.
/// * `replication` - which replication of the whole run, not of the chunk.
/// * `segment` - the segment's identifier.
/// * `year` - 0 for the left-truncated draw at the start of the run, and
///   `y + 1` for a replacement made in year `y`.
fn weibull_draw(key: [u64; 2], replication: u64, segment: usize, year: u64) -> f64 {
    draws::uniform_at(
        draws::index(draws::purpose::LIFETIMES, replication, segment as u64, year),
        key,
    )
}

/// One chunk of replications, per year and per segment class.
///
/// Every buffer holds `replications * years * classes` values, indexed
/// `(replication, year, class)` with class varying fastest, which is C order —
/// the layout NumPy reads them back in. The class axis is kept rather than
/// summed away because failures are read by class and a system total cannot be
/// decomposed afterwards.
///
/// The field names and their order match `simulate.Results`, the Python
/// NamedTuple the binding's caller rebuilds from these buffers.
pub struct Results {
    /// Segments that failed.
    pub failures: Vec<f64>,
    /// Customers out, counted per failure.
    pub customers_interrupted: Vec<f64>,
    /// Customer-minutes lost to failures.
    pub customer_minutes: Vec<f64>,
    /// Customer-minutes lost to planned work, which enters no reliability
    /// index and is reported on its own.
    pub planned_customer_minutes: Vec<f64>,
    /// Segments replaced as planned work.
    pub planned_replacements: Vec<f64>,
    /// Dollars spent on planned work, nominal.
    pub planned_spend: Vec<f64>,
    /// Dollars spent replacing failures, nominal.
    pub emergency_spend: Vec<f64>,
}

impl Results {
    /// Seven zeroed buffers of `replications * years * classes`.
    pub fn zeros(n_reps: usize, n_years: usize, n_classes: usize) -> Self {
        let cells = n_reps * n_years * n_classes;
        Self {
            failures: vec![0.0; cells],
            customers_interrupted: vec![0.0; cells],
            customer_minutes: vec![0.0; cells],
            planned_customer_minutes: vec![0.0; cells],
            planned_replacements: vec![0.0; cells],
            planned_spend: vec![0.0; cells],
            emergency_spend: vec![0.0; cells],
        }
    }

    /// Appends one replication's block to the end of every buffer.
    ///
    /// A block is a `Results` built with one replication, so its buffers are
    /// `years * classes` long and already in the layout this one wants next.
    /// Concatenating in replication order is what makes the parallel path
    /// return the same buffers as the sequential one: each worker computes a
    /// block that depends on nothing outside its own replication, and the
    /// order the blocks were *finished* in is discarded here.
    fn append(&mut self, block: Results) {
        self.failures.extend(block.failures);
        self.customers_interrupted
            .extend(block.customers_interrupted);
        self.customer_minutes.extend(block.customer_minutes);
        self.planned_customer_minutes
            .extend(block.planned_customer_minutes);
        self.planned_replacements.extend(block.planned_replacements);
        self.planned_spend.extend(block.planned_spend);
        self.emergency_spend.extend(block.emergency_spend);
    }
}

/// The working buffers one replication reads and overwrites.
///
/// Python allocates these fresh inside its replication loop and lets the
/// garbage collector reclaim last year's. Rust has no garbage collector, and at
/// the shipped population they are twelve thousand elements apiece over a
/// thirty-year horizon, so they are allocated once and overwritten instead.
///
/// Gathering them into a struct is what lets one body serve both the sequential
/// and the parallel path. Loose `let mut` bindings outside the loop cannot be
/// handed to a parallel iterator — several workers would hold `&mut` to the
/// same buffer at once, which the compiler refuses rather than allowing a data
/// race — so each worker builds its own and reuses it across the replications
/// it happens to be given.
struct Scratch<T: Real> {
    /// Current age of each segment, in years.
    age: Vec<T>,
    /// Weibull shape currently in the ground, which a replacement changes.
    current_shape: Vec<T>,
    /// Weibull scale currently in the ground.
    current_scale: Vec<T>,
    /// Simulation time at which each segment next fails, measured from year 0.
    failure_time: Vec<T>,
    /// Planned replacement cost at this year's prices.
    planned_now: Vec<T>,
    /// The rank key, written only for the segments that are candidates.
    rank: Vec<T>,
    /// Whether each segment has already been replaced this year.
    replaced: Vec<bool>,
    /// The eligible segments, ascending, refilled each year.
    candidates: Vec<usize>,
    /// The fixed per-segment priority the random policy ranks on.
    ///
    /// Filled once per replication rather than read per candidate per year. It
    /// does not depend on the year, so drawing it inside the year loop would
    /// recompute the same twelve thousand values once for every year of the
    /// horizon.
    priority: Vec<T>,
}

impl<T: Real> Scratch<T> {
    /// Buffers sized for one replication over `n_segments` segments.
    fn new(n_segments: usize) -> Self {
        Self {
            age: vec![T::zero(); n_segments],
            current_shape: vec![T::zero(); n_segments],
            current_scale: vec![T::zero(); n_segments],
            failure_time: vec![T::zero(); n_segments],
            planned_now: vec![T::zero(); n_segments],
            rank: vec![T::zero(); n_segments],
            replaced: vec![false; n_segments],
            candidates: Vec::with_capacity(n_segments),
            priority: vec![T::zero(); n_segments],
        }
    }
}

/// What running a chunk can fail with.
///
/// Rust has no exceptions: a function that can fail says so in its return type,
/// and the caller must handle both arms. Two unrelated things can go wrong
/// here, so they are gathered into one type with a variant apiece — the
/// equivalent of a Python function raising either of two exception classes,
/// except that the signature names them.
pub enum ChunkError {
    /// A candidate scored a rank key that is not a number.
    NanScore(policies::NanScore),
    /// The thread pool could not be built, so nothing ran.
    ThreadPool(rayon::ThreadPoolBuildError),
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkError::NanScore(nan) => write!(formatter, "{nan}"),
            ChunkError::ThreadPool(error) => write!(
                formatter,
                "could not build a thread pool for this chunk: {error}"
            ),
        }
    }
}

/// Runs one chunk of replications under one policy.
///
/// The arguments are the ones `simulate.run_chunk` takes, in that order and
/// under those names, so the two implementations are interchangeable behind a
/// keyword call. Every slice is C-contiguous, which the binding checks before
/// calling this.
///
/// # Arguments
///
/// * `length_ft` - segment length, in feet.
/// * `customers` - customers served, counted equally for the frequency index.
/// * `customer_minutes_per_failure` - customer-minutes lost when this segment
///   fails, already carrying its class's restoration time.
/// * `customer_minutes_per_planned` - the same for planned work, zero for a
///   class that is switched out without interrupting anyone.
/// * `outage_cost_per_failure` - value of lost load if this segment fails, in
///   dollars at year-0 prices.
/// * `class_index` - which segment class each segment belongs to, indexing the
///   third axis of the results.
/// * `age0` - age at the start of the run, in years.
/// * `shape` - Weibull shape for the cable in the ground, effective.
/// * `scale` - Weibull scale for the cable in the ground, effective.
/// * `replacement_shape` - Weibull shape a replacement would take, effective
///   for this segment's own geometry.
/// * `replacement_scale` - the same for scale.
/// * `cost_per_ft` - installed cost per foot.
/// * `draw_key` - the two key words every uniform is computed under. **No
///   draws are passed in.** A uniform here is a function of where it sits —
///   the purpose, the replication, the segment and the year — so a worker
///   produces the one it needs and coordinates with nobody, and the year draws
///   nobody reads are never produced.
/// * `first_replication` - where this chunk starts in the run. The replication
///   is part of a draw's address, so a chunk has to know where it sits or
///   splitting a run into chunks would change its numbers.
/// * `n_reps` - replications this chunk covers.
/// * `budget` - planned capital per year, already escalated.
/// * `cost_escalation` - per-year multiplier applied to every dollar quantity.
/// * `policy` - the resolved replacement policy.
/// * `emergency_multiplier` - what replacing a failure costs relative to the
///   same work planned.
/// * `mobilization_per_segment` - fixed cost of turning up at all.
/// * `emergency_charged_to_budget` - charge the year's emergency spend against
///   the planned budget before scoring planned work.
/// * `n_classes` - number of segment classes, sizing the third result axis.
/// * `n_years` - horizon, in years.
/// * `threads` - worker threads to spread the replications over. One runs them
///   sequentially with no thread pool built and no work-stealing at all, which
///   is what makes the single-threaded timing a baseline rather than a
///   measurement of rayon's overhead. Above one, a pool of that size is built
///   for this call and dropped with it, so the count is a property of the call
///   rather than of the process.
///
/// # Returns
///
/// The seven per-year, per-class buffers for this chunk, or a `ChunkError` if
/// a candidate scored a key that was not a number, or if the pool could not be
/// built.
///
/// The buffers do not depend on `threads`. Each replication computes its own
/// draws from their positions and writes its own block of the results, so no
/// two workers touch the same value, and the blocks are concatenated in
/// replication order rather than in the order they finished.
#[allow(clippy::too_many_arguments)]
pub fn run_chunk<T: Real>(
    length_ft: &[T],
    customers: &[T],
    customer_minutes_per_failure: &[T],
    customer_minutes_per_planned: &[T],
    outage_cost_per_failure: &[T],
    class_index: &[u8],
    age0: &[T],
    shape: &[T],
    scale: &[T],
    replacement_shape: &[T],
    replacement_scale: &[T],
    cost_per_ft: &[T],
    draw_key: [u64; 2],
    first_replication: u64,
    n_reps: usize,
    budget: &[T],
    cost_escalation: &[T],
    policy: Resolved,
    emergency_multiplier: f64,
    mobilization_per_segment: f64,
    emergency_charged_to_budget: bool,
    n_classes: usize,
    n_years: usize,
    threads: usize,
) -> Result<Results, ChunkError> {
    // The segment count is the population's; the replication count arrives as
    // `n_reps`, since nothing else passed in carries the replication axis.
    let n_segments = age0.len();
    // Configuration crosses from Python as a double whatever the run's
    // precision is, and is narrowed once here rather than at every use.
    let narrowed_multiplier = T::from_double(emergency_multiplier);

    // Costs at year-0 prices; the year's escalation is applied inside the loop.
    // `(0..n_segments).map(...).collect()` builds a list the way a Python
    // comprehension does; the type annotation on the binding is what tells
    // `collect` which kind of collection to build.
    let planned_at_par: Vec<T> = (0..n_segments)
        .map(|segment| {
            policies::planned_cost(
                length_ft[segment],
                cost_per_ft[segment],
                T::from_double(mobilization_per_segment),
            )
        })
        .collect();

    // One replication, start to finish, writing a block of its own.
    //
    // A closure rather than a named function so that it reads the arguments
    // above without taking all twenty-two of them again. Everything it touches
    // from outside is borrowed and never written, and that is exactly the
    // property that makes it safe to call from several threads at once —
    // checked when this compiles rather than trusted. A closure that wrote to
    // something it captured would not be accepted by the parallel branch below.
    //
    // The Python reference runs this same body as the outer level of a nested
    // loop. Splitting it out changes no arithmetic: a replication reads its own
    // slice of the draws and writes its own results, and shares nothing with
    // any other.
    let one_replication =
        |replication: usize, scratch: &mut Scratch<T>| -> Result<Results, policies::NanScore> {
            // Naming the fields once, rather than writing `scratch.` in front of
            // every use below. Each binding is a mutable borrow of one field, and
            // Rust tracks them separately, which is why several can be live at once.
            let Scratch {
                age,
                current_shape,
                current_scale,
                failure_time,
                planned_now,
                rank,
                replaced,
                candidates,
                priority,
            } = scratch;
            // This replication's own results: one replication's worth, so
            // `years * classes` per buffer.
            let mut block = Results::zeros(1, n_years, n_classes);

            age.copy_from_slice(age0);
            current_shape.copy_from_slice(shape);
            current_scale.copy_from_slice(scale);

            // Where this replication sits in the whole run, which is part of every
            // draw's address. Using the position within the chunk instead would
            // make the chunking change the numbers.
            let replication_in_run = first_replication + replication as u64;

            // The two draws every segment takes whatever happens to it, each a run
            // of consecutive positions and so a quarter as many encryptions as
            // asking for them one at a time. The priority does not depend on the
            // year, so it is filled here rather than inside the year loop.
            draws::fill_run(
                draws::index(draws::purpose::POLICIES, replication_in_run, 0, 0),
                draw_key,
                priority,
            );
            draws::fill_run(
                draws::index(draws::purpose::LIFETIMES, replication_in_run, 0, 0),
                draw_key,
                failure_time,
            );

            // Conditional on survival to age0: a population that starts partway
            // through its life must not behave as though it were new.
            for segment in 0..n_segments {
                failure_time[segment] = weibull::draw_remaining_life(
                    failure_time[segment],
                    age[segment],
                    current_shape[segment],
                    current_scale[segment],
                );
            }

            for year in 0..n_years {
                let escalation = cost_escalation[year];
                for segment in 0..n_segments {
                    planned_now[segment] = planned_at_par[segment] * escalation;
                }
                // Where this year's totals start in each buffer; the segment's
                // class is added to it to reach the exact cell. The replication
                // does not appear because this block holds one.
                let cell = year * n_classes;

                // 1. Failures, which are resolved before planned work so that a
                //    segment failing this year is not also a candidate this year.
                // Reused rather than reallocated, so it has to be cleared. The
                // reference allocates a fresh boolean array each year instead.
                replaced.fill(false);
                // In the working width, not in double: its one use is subtracted
                // from the budget the greedy fill then compares a cumulative cost
                // against, so it has to round the way that comparison rounds.
                // NumPy's `cumsum` preserves the width for the same reason, where
                // its `bincount` widens the result totals to double.
                let mut emergency_total = T::zero();
                let year_start = T::from_double(year as f64);
                for segment in 0..n_segments {
                    if failure_time[segment] >= year_start
                        && failure_time[segment] < year_start + T::one()
                    {
                        let emergency_now = planned_now[segment] * narrowed_multiplier;
                        // `usize::from` widens the `u8` class index without a
                        // cast that could silently lose bits; Rust will not add a
                        // `u8` to a `usize` on its own.
                        let class = cell + usize::from(class_index[segment]);
                        block.failures[class] += 1.0;
                        block.customers_interrupted[class] += customers[segment].into_double();
                        block.customer_minutes[class] +=
                            customer_minutes_per_failure[segment].into_double();
                        block.emergency_spend[class] += emergency_now.into_double();
                        emergency_total = emergency_total + emergency_now;
                        replaced[segment] = true;
                    }
                }

                // 2. Planned replacement, funded greedily down the ranked order.
                let mut available = budget[year];
                if emergency_charged_to_budget {
                    // Charged before this year's planned pass is scored, which is
                    // what produces the loop where failures crowd out prevention.
                    //
                    // A year whose failures cost more than the budget leaves this
                    // negative, and that is left alone rather than floored at zero.
                    // Every planned cost is positive, so the greedy fill funds
                    // nothing at any value at or below zero, and nothing carries to
                    // the next year — each year takes the amount in the budget
                    // series and no more. A floor here would be a line no result
                    // could distinguish from its absence, which mutation testing
                    // confirms: removing one left the whole suite green.
                    // `Float` does not require `SubAssign`.
                    available = available - emergency_total;
                }

                candidates.clear();
                for segment in 0..n_segments {
                    if policies::eligible(policy, age[segment], replaced[segment]) {
                        candidates.push(segment);
                    }
                }
                if !candidates.is_empty() {
                    // Only the candidates are scored. The reference scores every
                    // segment because that is what vectorizes, and then reads the
                    // candidates' entries; the values it computes for the rest are
                    // never read, so the two agree on everything either one uses.
                    for &segment in candidates.iter() {
                        rank[segment] = policies::rank_key(
                            policy,
                            age[segment],
                            weibull::conditional_failure_probability(
                                age[segment],
                                current_shape[segment],
                                current_scale[segment],
                            ),
                            outage_cost_per_failure[segment] * escalation,
                            planned_now[segment],
                            narrowed_multiplier,
                            priority[segment],
                        );
                    }
                    // The `?` returns early with the error if there was one, and
                    // unwraps the value otherwise. It is Rust's substitute for an
                    // exception propagating up the stack: the failure travels the
                    // same way, but every function it passes through has to name
                    // it in its return type.
                    let ordered = policies::order_by_rank(rank, candidates)?;
                    for &segment in policies::fund(&ordered, planned_now, available) {
                        let class = cell + usize::from(class_index[segment]);
                        block.planned_replacements[class] += 1.0;
                        block.planned_customer_minutes[class] +=
                            customer_minutes_per_planned[segment].into_double();
                        block.planned_spend[class] += planned_now[segment].into_double();
                        replaced[segment] = true;
                    }
                }

                // 3. Everything replaced this year enters service next year, as new
                //    cable of the replacement technology, with a lifetime drawn
                //    from that segment's cell for the following year.
                // The reference writes `age += 1.0` over the whole array and then
                // `age[replaced] = 0.0`, which is two passes because that is what
                // vectorizes. One pass with a branch is the same result.
                for segment in 0..n_segments {
                    if replaced[segment] {
                        current_shape[segment] = replacement_shape[segment];
                        current_scale[segment] = replacement_scale[segment];
                        failure_time[segment] = T::from_double((year + 1) as f64)
                            + weibull::draw_lifetime(
                                T::from_double(weibull_draw(
                                    draw_key,
                                    replication_in_run,
                                    segment,
                                    year as u64 + 1,
                                )),
                                replacement_shape[segment],
                                replacement_scale[segment],
                            );
                        age[segment] = T::zero();
                    } else {
                        age[segment] = age[segment] + T::one();
                    }
                }
            }
            Ok(block)
        };

    let blocks: Vec<Result<Results, policies::NanScore>> = if threads <= 1 {
        // No pool, no work-stealing, no atomics. This path is the baseline the
        // parallel one is measured against, so it must not pay for machinery it
        // does not use, and one `Scratch` serves every replication in turn.
        let mut scratch = Scratch::new(n_segments);
        (0..n_reps)
            .map(|replication| one_replication(replication, &mut scratch))
            .collect()
    } else {
        // A pool built for this call and dropped with it, rather than rayon's
        // process-wide default: the thread count is an argument here, and the
        // global pool can only be sized once in the life of a process, so a
        // benchmark timing several thread counts in one run could not use it.
        //
        // `map_init` gives each worker its own `Scratch`, built the first time
        // that worker is handed a replication and reused for every one after —
        // the parallel equivalent of allocating once outside the loop.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(ChunkError::ThreadPool)?;
        pool.install(|| {
            (0..n_reps)
                .into_par_iter()
                .map_init(
                    || Scratch::new(n_segments),
                    |scratch, replication| one_replication(replication, scratch),
                )
                .collect()
        })
    };

    // One outcome per replication, folded here rather than short-circuited
    // inside the iterator. A parallel run that stopped at the first failure
    // would stop at whichever one a worker reached first, so the same bad input
    // could name a different segment from run to run; collecting every outcome
    // and reporting the earliest replication's makes the message reproducible.
    let mut results = Results::zeros(0, n_years, n_classes);
    for block in blocks {
        results.append(block.map_err(ChunkError::NanScore)?);
    }
    Ok(results)
}
