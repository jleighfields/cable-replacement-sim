"""The annual loop compiled by Numba, in the shape the scalar reference has.

This is a third implementation of the model, and it exists to separate two
things the benchmark table cannot currently tell apart: whether the Rust
kernel's advantage comes from Rust, or from being compiled and threaded at all.
It computes what the reference computes, cell for cell, and is held to that
exactly rather than within a tolerance.

**It follows the scalar reference rather than the batched loop**, and that is
the point of it. Vectorising costs the ability to skip: the batched
implementation sorts every segment in every replication because that is what a
whole-array sort does, while the reference sorts only the candidates a policy
made eligible. A compiled scalar loop keeps the skip and pays no interpreter
cost for it, which is the combination neither existing Python implementation
offers.

Three things about reading this beside the other implementations:

* **Every constant is typed ``uint64``.** Numba follows NumPy's promotion
  rules, so a ``uint64`` combined with an untyped integer literal produces a
  ``float64``, and the generator would go on producing plausible numbers that
  are not the right ones. The values themselves come from ``random_draws``,
  which reads them from the compiled crate, so nothing here is a second copy.
* **The 64-by-64 product is taken in halves.** Numba has no 128-bit integer,
  where the Rust side uses ``u128`` and Python multiplies in arbitrary
  precision.
* **Sums that decide a discrete outcome accumulate left to right.** A pairwise
  sum disagrees in the last bits, and where the result is compared against a
  budget that difference changes which segment is funded last.
"""

import math

import numba
import numpy as np

from cablesim import policies, random_draws, simulate

# Every value below is read from `random_draws`, which reads the ones that
# cross the language boundary from the compiled crate. Numba freezes a global
# at compile time, so these become constants in the generated code while still
# having exactly one definition in the project.
MULTIPLIER_0 = np.uint64(random_draws.PHILOX_MULTIPLIERS[0])
MULTIPLIER_1 = np.uint64(random_draws.PHILOX_MULTIPLIERS[1])
WEYL_0 = np.uint64(random_draws.PHILOX_WEYL[0])
WEYL_1 = np.uint64(random_draws.PHILOX_WEYL[1])
PHILOX_ROUNDS = random_draws.PHILOX_ROUNDS
LANES = np.uint64(random_draws.PHILOX_LANES)
MANTISSA_SHIFT = np.uint64(random_draws.MANTISSA_SHIFT)
TWO_TO_THE_FIFTY_THIRD = float(random_draws.TWO_TO_THE_FIFTY_THIRD)

PURPOSE_SHIFT = np.uint64(random_draws.PURPOSE_SHIFT)
REPLICATION_SHIFT = np.uint64(random_draws.REPLICATION_SHIFT)
YEAR_SHIFT = np.uint64(random_draws.YEAR_SHIFT)
SEGMENT_SHIFT = np.uint64(random_draws.SEGMENT_SHIFT)
PURPOSE_LIFETIMES = np.uint64(random_draws.PURPOSE["lifetimes"])
PURPOSE_POLICIES = np.uint64(random_draws.PURPOSE["policies"])

KIND_AGE_THRESHOLD = policies.KIND["age_threshold"]
KIND_RISK_RANKED = policies.KIND["risk_ranked"]
KIND_WORST_FIRST = policies.KIND["worst_first"]
KIND_RANDOM = policies.KIND["random"]

MASK_32 = np.uint64(0xFFFFFFFF)
HALF_WORD = np.uint64(32)
ONE = np.uint64(1)
ZERO = np.uint64(0)


@numba.njit(cache=True, inline="always")
def multiply_wide(left, right):
    """Both halves of a 64-by-64 bit product, built from 32-bit pieces.

    Args:
        left: One factor.
        right: The other factor.

    Returns:
        The high and low 64-bit halves of the product.
    """
    left_low = left & MASK_32
    left_high = left >> HALF_WORD
    right_low = right & MASK_32
    right_high = right >> HALF_WORD

    low_low = left_low * right_low
    high_low = left_high * right_low
    low_high = left_low * right_high
    high_high = left_high * right_high

    cross = (low_low >> HALF_WORD) + (high_low & MASK_32) + low_high
    high = high_high + (cross >> HALF_WORD) + (high_low >> HALF_WORD)
    low = (cross << HALF_WORD) | (low_low & MASK_32)
    return high, low


@numba.njit(cache=True)
def philox(counter_0, counter_1, counter_2, counter_3, key_0, key_1):
    """Encrypts one counter under one key, giving four uniform words.

    The whole generator. It carries no state between calls, which is what makes
    it safe to call from any thread at any position.

    Args:
        counter_0: First word of the counter being encrypted.
        counter_1: Second word.
        counter_2: Third word.
        counter_3: Fourth word.
        key_0: Low key word.
        key_1: High key word.

    Returns:
        Four uniformly distributed 64-bit words.
    """
    state_0, state_1 = counter_0, counter_1
    state_2, state_3 = counter_2, counter_3
    schedule_0, schedule_1 = key_0, key_1
    for index in range(PHILOX_ROUNDS):
        if index > 0:
            schedule_0 = schedule_0 + WEYL_0
            schedule_1 = schedule_1 + WEYL_1
        high_0, low_0 = multiply_wide(MULTIPLIER_0, state_0)
        high_1, low_1 = multiply_wide(MULTIPLIER_1, state_2)
        state_0 = high_1 ^ state_1 ^ schedule_0
        state_1 = low_1
        state_2 = high_0 ^ state_3 ^ schedule_1
        state_3 = low_0
    return state_0, state_1, state_2, state_3


@numba.njit(cache=True, inline="always")
def to_double(word):
    """Converts one uniform word to a double in ``[0, 1)``.

    The top 53 bits are kept, which is every bit a double represents without
    rounding. This has to be NumPy's conversion exactly, or the implementations
    disagree in the last bit of every draw.

    Args:
        word: A uniform 64-bit word.

    Returns:
        A double in ``[0, 1)``.
    """
    return np.float64(word >> MANTISSA_SHIFT) * TWO_TO_THE_FIFTY_THIRD


@numba.njit(cache=True, inline="always")
def draw_index(purpose, replication, segment, year):
    """Packs a draw's four coordinates into the position it is computed at.

    Args:
        purpose: Which stream, keeping the uses of randomness uncorrelated.
        replication: Which replication.
        segment: Which segment.
        year: Which year.

    Returns:
        The flat draw index.
    """
    return (
        (purpose << PURPOSE_SHIFT)
        | (replication << REPLICATION_SHIFT)
        | (year << YEAR_SHIFT)
        | (segment << SEGMENT_SHIFT)
    )


@numba.njit(cache=True, inline="always")
def uniform_at(index, key_0, key_1):
    """The one uniform at a draw index.

    NumPy increments its counter before producing, so the block it emits at
    counter ``b`` is the one Philox defines at ``b + 1``; that offset is
    applied here so no caller carries it.

    Args:
        index: The flat draw index, from ``draw_index``.
        key_0: Low key word.
        key_1: High key word.

    Returns:
        A double in ``[0, 1)``.
    """
    words = philox(index // LANES + ONE, ZERO, ZERO, ZERO, key_0, key_1)
    lane = index % LANES
    if lane == ZERO:
        return to_double(words[0])
    elif lane == ONE:
        return to_double(words[1])
    elif lane == np.uint64(2):
        return to_double(words[2])
    else:
        return to_double(words[3])


@numba.njit(cache=True, inline="always")
def conditional_failure_probability(age, shape, scale):
    """Probability of failing within the year, given survival to ``age``.

    Args:
        age: Current age, in years.
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        The probability, in ``[0, 1)``.
    """
    annual_hazard = ((age + 1.0) / scale) ** shape - (age / scale) ** shape
    return -math.expm1(-annual_hazard)


@numba.njit(cache=True, inline="always")
def draw_lifetime(uniform, shape, scale):
    """Samples a lifetime for new cable.

    Args:
        uniform: A uniform on ``[0, 1)``.
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        The lifetime, in years.
    """
    return scale * (-math.log1p(-uniform)) ** (1.0 / shape)


@numba.njit(cache=True, inline="always")
def draw_remaining_life(uniform, age, shape, scale):
    """Samples remaining life for cable that has already survived to ``age``.

    Conditional on that survival, which adds the hazard already accumulated
    back in. Drawing unconditionally makes a population that starts partway
    through its life behave as though it were new, and does so silently.

    Args:
        uniform: A uniform on ``[0, 1)``.
        age: Current age, in years.
        shape: Weibull shape parameter.
        scale: Effective Weibull scale, in years.

    Returns:
        Remaining life from ``age``, in years.
    """
    accumulated = (age / scale) ** shape
    total = scale * (accumulated - math.log1p(-uniform)) ** (1.0 / shape)
    return total - age


@numba.njit(cache=True)
def order_by_rank(scores, candidates, order):
    """Orders candidates by score descending, breaking ties on segment.

    The key is total, so a tie cannot decide the answer. Ties arise constantly
    — an age threshold ranks on an integer age thousands of segments share —
    and the greedy fill stops at the first candidate that does not fit, so
    which tied segment lands last decides whether it is funded.

    ``candidates`` arrives ascending, so a **stable** sort on the negated score
    produces exactly the pair the other implementations sort on.

    Args:
        scores: The score of each candidate, positionally matched.
        candidates: Eligible segment identifiers, ascending.
        order: Scratch of at least ``candidates.size``, written with the result.

    Returns:
        How many entries of ``order`` were written.
    """
    count = candidates.size
    positions = np.argsort(-scores[:count], kind="mergesort")
    for slot in range(count):
        order[slot] = candidates[positions[slot]]
    return count


@numba.njit(cache=True)
def run_replication(
    replication,
    first_replication,
    key_0,
    key_1,
    age0,
    shape,
    scale,
    replacement_shape,
    replacement_scale,
    customers,
    customer_minutes_per_failure,
    customer_minutes_per_planned,
    outage_cost_per_failure,
    class_index,
    planned_at_par,
    budget,
    cost_escalation,
    policy_kind,
    threshold_years,
    rank_by_cost,
    emergency_multiplier,
    emergency_charged_to_budget,
    n_years,
    failures,
    customers_interrupted,
    customer_minutes,
    planned_customer_minutes,
    planned_replacements,
    planned_spend,
    emergency_spend,
):
    """Runs one replication, writing its rows of every result array.

    Split out from the chunk loop so the parallel and serial forms share one
    body: whichever way the replications are handed out, each one computes its
    own draws from their positions and touches only its own rows, so nothing is
    shared and the answer cannot depend on the thread count.

    Args:
        replication: Which replication within this chunk.
        first_replication: Where this chunk starts in the run, which the draw
            index is built from so a chunk gives the same numbers wherever it
            sits.
        key_0: Low key word.
        key_1: High key word.
        age0: Age of each segment at the start, in years.
        shape: Weibull shape for the cable in the ground.
        scale: Effective Weibull scale for the cable in the ground.
        replacement_shape: Weibull shape for replacement cable.
        replacement_scale: Effective Weibull scale for replacement cable.
        customers: Customers served by each segment.
        customer_minutes_per_failure: Customer-minutes lost per failure.
        customer_minutes_per_planned: Customer-minutes lost per planned job.
        outage_cost_per_failure: Value of lost load per failure, in dollars.
        class_index: Which reporting class each segment belongs to.
        planned_at_par: Planned replacement cost at year-zero prices.
        budget: Dollars available in each year.
        cost_escalation: Price level in each year, relative to year zero.
        policy_kind: The policy's integer tag.
        threshold_years: Eligibility age, compared as ``age >= threshold``.
        rank_by_cost: Whether to divide the score by planned cost.
        emergency_multiplier: Emergency cost relative to the same work planned.
        emergency_charged_to_budget: Whether failures are charged to the budget
            before the year's planned pass is scored.
        n_years: Years in the horizon.
        failures: Result array, written at this replication's rows.
        customers_interrupted: Result array, likewise.
        customer_minutes: Result array, likewise.
        planned_customer_minutes: Result array, likewise.
        planned_replacements: Result array, likewise.
        planned_spend: Result array, likewise.
        emergency_spend: Result array, likewise.
    """
    n_segments = age0.size
    # Both cast before adding: a uint64 combined with the int64 the loop
    # counter is would promote the sum to float64 under NumPy's rules.
    in_run = np.uint64(first_replication) + np.uint64(replication)

    age = age0.astype(np.float64)
    current_shape = shape.astype(np.float64)
    current_scale = scale.astype(np.float64)
    failure_time = np.empty(n_segments, np.float64)
    priority = np.empty(n_segments, np.float64)

    # The two draws every segment takes whatever happens to it. Both are read
    # for every segment, so there is nothing to select and they are taken up
    # front, conditional on survival to age0 in the lifetime's case.
    for segment in range(n_segments):
        position = np.uint64(segment)
        priority[segment] = uniform_at(
            draw_index(PURPOSE_POLICIES, in_run, position, ZERO), key_0, key_1
        )
        initial = uniform_at(
            draw_index(PURPOSE_LIFETIMES, in_run, position, ZERO), key_0, key_1
        )
        failure_time[segment] = draw_remaining_life(
            initial, age[segment], current_shape[segment], current_scale[segment]
        )

    replaced = np.empty(n_segments, np.bool_)
    candidates = np.empty(n_segments, np.int64)
    scores = np.empty(n_segments, np.float64)
    order = np.empty(n_segments, np.int64)
    planned_now = np.empty(n_segments, np.float64)

    for year in range(n_years):
        escalation = cost_escalation[year]
        for segment in range(n_segments):
            planned_now[segment] = planned_at_par[segment] * escalation
            replaced[segment] = False

        # 1. Failures, resolved before planned work so that a segment failing
        #    this year is not also a candidate this year.
        emergency_running = 0.0
        for segment in range(n_segments):
            if failure_time[segment] >= year and failure_time[segment] < year + 1:
                replaced[segment] = True
                bin_index = class_index[segment]
                emergency_now = planned_now[segment] * emergency_multiplier
                failures[replication, year, bin_index] += 1.0
                customers_interrupted[replication, year, bin_index] += customers[
                    segment
                ]
                customer_minutes[replication, year, bin_index] += (
                    customer_minutes_per_failure[segment]
                )
                emergency_spend[replication, year, bin_index] += emergency_now
                # Left to right, because this total is subtracted from the
                # budget the greedy fill then compares a cumulative cost
                # against: a last-bit difference decides which segment is
                # funded last.
                emergency_running += emergency_now

        # 2. Planned replacement, funded greedily down the ranked order.
        available = budget[year]
        if emergency_charged_to_budget:
            # A year whose failures cost more than the budget leaves this
            # negative, which is left alone rather than floored: every planned
            # cost is positive, so the fill funds nothing at or below zero and
            # nothing carries into the next year.
            available -= emergency_running

        count = 0
        for segment in range(n_segments):
            if age[segment] >= threshold_years and not replaced[segment]:
                candidates[count] = segment
                if policy_kind == KIND_AGE_THRESHOLD:
                    key = age[segment]
                elif policy_kind == KIND_RISK_RANKED:
                    # The premium avoided by acting first, rather than the whole
                    # emergency cost: the planned work is paid either way.
                    avoided = planned_now[segment] * (emergency_multiplier - 1.0)
                    probability = conditional_failure_probability(
                        age[segment], current_shape[segment], current_scale[segment]
                    )
                    key = probability * (
                        outage_cost_per_failure[segment] * escalation + avoided
                    )
                elif policy_kind == KIND_WORST_FIRST:
                    key = conditional_failure_probability(
                        age[segment], current_shape[segment], current_scale[segment]
                    )
                elif policy_kind == KIND_RANDOM:
                    key = priority[segment]
                else:
                    key = 0.0
                if rank_by_cost:
                    # Planned cost, because that is what the budget is charged,
                    # so the ratio is value per budget dollar.
                    key = key / planned_now[segment]
                if np.isnan(key):
                    raise ValueError(
                        "a candidate segment scored NaN; every input to the "
                        "score is finite by construction, so this is a defect "
                        "upstream of ranking"
                    )
                scores[count] = key
                count += 1

        if count > 0:
            order_by_rank(scores[:count], candidates[:count], order)
            # Stops at the first candidate that does not fit, rather than
            # passing over it to fund cheaper ones below: that alternative is a
            # knapsack heuristic and is inherently sequential, so no vectorized
            # implementation could reproduce it.
            spent = 0.0
            for slot in range(count):
                segment = order[slot]
                spent += planned_now[segment]
                if spent > available:
                    break
                bin_index = class_index[segment]
                planned_replacements[replication, year, bin_index] += 1.0
                planned_customer_minutes[replication, year, bin_index] += (
                    customer_minutes_per_planned[segment]
                )
                planned_spend[replication, year, bin_index] += planned_now[segment]
                replaced[segment] = True

        # 3. Everything replaced this year enters service next year, as new
        #    cable of the replacement technology.
        for segment in range(n_segments):
            if replaced[segment]:
                current_shape[segment] = replacement_shape[segment]
                current_scale[segment] = replacement_scale[segment]
                renewal = uniform_at(
                    draw_index(
                        PURPOSE_LIFETIMES,
                        in_run,
                        np.uint64(segment),
                        np.uint64(year + 1),
                    ),
                    key_0,
                    key_1,
                )
                failure_time[segment] = (year + 1) + draw_lifetime(
                    renewal,
                    replacement_shape[segment],
                    replacement_scale[segment],
                )
                age[segment] = 0.0
            else:
                age[segment] += 1.0


@numba.njit(cache=True, parallel=True)
def run_chunk_parallel(
    n_reps,
    first_replication,
    key_0,
    key_1,
    age0,
    shape,
    scale,
    replacement_shape,
    replacement_scale,
    customers,
    customer_minutes_per_failure,
    customer_minutes_per_planned,
    outage_cost_per_failure,
    class_index,
    planned_at_par,
    budget,
    cost_escalation,
    policy_kind,
    threshold_years,
    rank_by_cost,
    emergency_multiplier,
    emergency_charged_to_budget,
    n_years,
    failures,
    customers_interrupted,
    customer_minutes,
    planned_customer_minutes,
    planned_replacements,
    planned_spend,
    emergency_spend,
):
    """Spreads the replications of one chunk over Numba's threads.

    Each replication computes its own draws from their positions and writes
    only its own rows, so the workers share nothing and no accumulation crosses
    them. The answer does not depend on how many threads run it, which is the
    property the thread-count test asserts.

    Args are those of ``run_replication``, with ``n_reps`` replacing the single
    replication index.
    """
    for replication in numba.prange(n_reps):
        run_replication(
            replication,
            first_replication,
            key_0,
            key_1,
            age0,
            shape,
            scale,
            replacement_shape,
            replacement_scale,
            customers,
            customer_minutes_per_failure,
            customer_minutes_per_planned,
            outage_cost_per_failure,
            class_index,
            planned_at_par,
            budget,
            cost_escalation,
            policy_kind,
            threshold_years,
            rank_by_cost,
            emergency_multiplier,
            emergency_charged_to_budget,
            n_years,
            failures,
            customers_interrupted,
            customer_minutes,
            planned_customer_minutes,
            planned_replacements,
            planned_spend,
            emergency_spend,
        )


def run_chunk_numba(
    length_ft: np.ndarray,
    customers: np.ndarray,
    customer_minutes_per_failure: np.ndarray,
    customer_minutes_per_planned: np.ndarray,
    outage_cost_per_failure: np.ndarray,
    class_index: np.ndarray,
    age0: np.ndarray,
    shape: np.ndarray,
    scale: np.ndarray,
    replacement_shape: np.ndarray,
    replacement_scale: np.ndarray,
    cost_per_ft: np.ndarray,
    draw_key: tuple[int, int],
    first_replication: int,
    n_reps: int,
    budget: np.ndarray,
    cost_escalation: np.ndarray,
    policy: policies.Resolved,
    emergency_multiplier: float,
    mobilization_per_segment: float,
    emergency_charged_to_budget: bool,
    n_classes: int,
    n_years: int,
    threads: int = 1,
) -> simulate.Results:
    """Runs one chunk of replications under one policy, compiled by Numba.

    Takes the arguments every implementation of the annual loop takes, so it
    can be named at the call wherever the others can.

    Args:
        length_ft: Segment length, in feet.
        customers: Customers served, counted equally for the frequency index.
        customer_minutes_per_failure: Customer-minutes lost per failure.
        customer_minutes_per_planned: The same for planned work.
        outage_cost_per_failure: Value of lost load per failure, in dollars.
        class_index: Reporting class of each segment.
        age0: Age of each segment at the start, in years.
        shape: Weibull shape for the cable in the ground, effective.
        scale: Weibull scale for the cable in the ground, effective.
        replacement_shape: The same for replacement cable.
        replacement_scale: The same for scale.
        cost_per_ft: Installed cost per foot.
        draw_key: The two key words, from ``random_draws.draw_key``.
        first_replication: Where this chunk starts in the run.
        n_reps: Replications in this chunk.
        budget: Dollars available in each year.
        cost_escalation: Price level in each year, relative to year zero.
        policy: The resolved policy.
        emergency_multiplier: Emergency cost relative to the same work planned.
        mobilization_per_segment: Fixed cost of turning up at all.
        emergency_charged_to_budget: Whether failures are charged to the budget.
        n_classes: Reporting classes.
        n_years: Years in the horizon.
        threads: Replications to run at once. One runs the serial form.

    Returns:
        The seven per-year, per-class arrays for this chunk.
    """
    n_segments = simulate.check_arguments(locals(), concurrent=True)
    results = simulate.Results(
        *(np.zeros((n_reps, n_years, n_classes)) for _ in simulate.Results._fields)
    )
    planned_at_par = policies.planned_cost(
        length_ft, cost_per_ft, mobilization_per_segment
    )
    arguments = (
        np.uint64(first_replication),
        np.uint64(draw_key[0]),
        np.uint64(draw_key[1]),
        np.ascontiguousarray(age0, dtype=np.float64),
        np.ascontiguousarray(shape, dtype=np.float64),
        np.ascontiguousarray(scale, dtype=np.float64),
        np.ascontiguousarray(replacement_shape, dtype=np.float64),
        np.ascontiguousarray(replacement_scale, dtype=np.float64),
        np.ascontiguousarray(customers, dtype=np.float64),
        np.ascontiguousarray(customer_minutes_per_failure, dtype=np.float64),
        np.ascontiguousarray(customer_minutes_per_planned, dtype=np.float64),
        np.ascontiguousarray(outage_cost_per_failure, dtype=np.float64),
        np.ascontiguousarray(class_index, dtype=np.int64),
        np.ascontiguousarray(planned_at_par, dtype=np.float64),
        np.ascontiguousarray(budget, dtype=np.float64),
        np.ascontiguousarray(cost_escalation, dtype=np.float64),
        int(policy.kind),
        float(policy.threshold_years),
        bool(policy.rank_by_cost),
        float(emergency_multiplier),
        bool(emergency_charged_to_budget),
        int(n_years),
        *results,
    )
    del n_segments

    if threads == 1:
        for replication in range(n_reps):
            run_replication(replication, *arguments)
    else:
        numba.set_num_threads(threads)
        run_chunk_parallel(n_reps, *arguments)
    return results
