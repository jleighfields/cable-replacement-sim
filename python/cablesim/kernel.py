"""The Rust compute kernel, behind the reference's own call signature.

`simulate.py` is the correctness reference and this is the fast path, and the
point of this module is that nothing above them can tell which it holds. The
call takes the same twenty-four arguments under the same names and returns the
same `simulate.Results`, so `run.py`, the metrics layer and every test drive
both through one path rather than through two that could differ in how they are
driven. `threads` is one of those arguments on both sides: the reference accepts
it and refuses anything but one, which says what it cannot do rather than
silently ignoring the request.

**No draw array crosses this boundary.** The kernel computes each uniform from
its position — the purpose, the replication, the segment and the year — under a
key derived from the run's seed, so a worker produces what it needs without a
stream to share or an array to be handed. `random_draws` computes the identical
values in NumPy for the implementations that stay in Python, and the two are
held to agreeing bit for bit.

The extension module returns seven arrays rather than a result object of its
own. Rebuilding the reference's NamedTuple from them here is what keeps a
single result type in the package: anything that unpacks a result, reads
`_fields`, or indexes it would otherwise work on one implementation and fail on
the other.

Build it before using it — `uv run maturin develop --release` — or this imports
whichever extension module was installed last rather than the one in the
working tree.
"""

import numpy as np

from cablesim import _cablesim, policies, simulate

AVAILABLE_THREADS: int = _cablesim.available_threads()
"""How many threads this machine can run at once.

Read from the extension module rather than from Python's own processor count,
because the number here and the number rayon would choose for itself have to be
the same one: it honours a CPU affinity mask and a container's CPU quota, and a
bare processor count does not. Callers wanting every core pass this rather than
a sentinel, so what a run records is the number that actually ran.
"""

BUILD_PROFILE: str = _cablesim.BUILD_PROFILE
"""Which Cargo profile the loaded extension module was compiled with.

Reported by the compiled binary rather than assumed by its caller, so a run's
recorded provenance cannot disagree with what produced it. Anything recording a
timing has to record this beside it: a debug build is slower by a wide enough
margin to make an unlabelled measurement meaningless.
"""


def run_chunk(
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
    """Runs one chunk of replications under one policy, in Rust.

    The arguments are `simulate.run_chunk`'s, and mean the same things. Every
    array crosses the boundary as the bytes NumPy already holds rather than as
    a copy. The two implementations read identical draws because each computes
    them from the same key and the same positions, which `tests/test_draws.py`
    establishes; there is no random-number stream to reconcile.

    **The dtype of the float arrays is the working precision, and it selects
    which entry point of the crate is called.** ``float32`` arrays reach the
    single-precision one and ``float64`` arrays the double-precision one;
    ``age0`` is what is read to decide, and every other float array has to
    carry the same dtype or the extraction refuses it by name. Nothing is
    converted, because converting would allocate a copy of every per-segment
    array on every call.

    Args:
        length_ft: Segment length, in feet.
        customers: Customers served, counted equally for the frequency index.
        customer_minutes_per_failure: Customer-minutes lost when this segment
            fails, already carrying its class's restoration time.
        customer_minutes_per_planned: The same for planned work, zero for a
            class that is switched out without interrupting anyone.
        outage_cost_per_failure: Value of lost load if this segment fails, in
            dollars at year-0 prices.
        class_index: Which segment class each segment belongs to, as
            ``uint8``, indexing the third axis of the returned arrays.
        age0: Age at the start of the run, in years.
        shape: Weibull shape for the cable in the ground, effective.
        scale: Weibull scale for the cable in the ground, effective.
        replacement_shape: Weibull shape a replacement would take, effective
            for this segment's own geometry.
        replacement_scale: The same for scale.
        cost_per_ft: Installed cost per foot.
        draw_key: The two key words every uniform is computed under. **The
            kernel produces its own draws from these**, which is why no array
            of them crosses the boundary: a draw is a function of where it
            sits, so a worker computes the one it needs and coordinates with
            nobody. The Python implementations compute the identical values
            from the identical key, which the draw tests establish.
        first_replication: Where this chunk starts in the run, so splitting a
            run into chunks changes no number.
        n_reps: Replications this chunk covers.
        budget: Planned capital per year, already escalated.
        cost_escalation: Per-year multiplier applied to every dollar quantity.
        policy: The resolved replacement policy.
        emergency_multiplier: What replacing a failure costs relative to the
            same work planned.
        mobilization_per_segment: Fixed cost of turning up at all.
        emergency_charged_to_budget: Charge the year's emergency spend against
            the planned budget before scoring planned work.
        n_classes: Number of segment classes, sizing the third result axis.
        n_years: Horizon, in years.
        threads: Workers to spread the replications over, which changes no
            number: a replication computes its own draws from their positions
            and writes its own block of the results, so the answer does not
            depend on how many workers there were or on the order they finished
            in. One runs them in sequence and builds no thread pool at all,
            which is what makes a single-threaded timing a baseline rather than
            a measurement of the pool's overhead. ``AVAILABLE_THREADS`` is this
            machine's count.

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: If the policy tag names no policy, if the population is
            empty, if ``n_classes`` is 0, if a class index is past the end of
            the class axis, if an array is not C-contiguous, if a per-segment
            or per-year array is the wrong length, if a replication, segment or
            year this chunk would draw at is past what a draw index can carry,
            if ``threads`` is 0, or if a candidate scores a rank key that is
            not a number.
        RuntimeError: If a thread pool of the requested size could not be
            built. That is the operating system refusing to start the threads
            rather than anything about the arguments, so it is not a
            ``ValueError``, and it has no counterpart in the reference.
        TypeError: If an array's dtype is not the one the boundary reads —
            ``uint8`` for ``class_index``, and for the rest whichever of
            ``float64`` and ``float32`` ``age0`` carries — if an array has the
            wrong number of axes, or if ``policy.kind`` is not an integer.
            The binding does not convert, because converting would allocate a
            copy of every per-segment array on every call.
            These come from PyO3's extraction rather than from any check here,
            so the messages are its wording; ``simulate.run_chunk`` raises the
            same class for the same inputs.
    """
    # The crate exposes one entry point per width, because PyO3 generates its
    # glue from a signature and a signature names a concrete element type. The
    # dtype of the arrays is what the run's precision travels as, so reading it
    # here is what turns that choice into a call — and a mismatched dtype
    # raises from the generated glue naming the array, rather than being
    # coerced into a converted copy of every per-segment array.
    at_width = (
        _cablesim.run_chunk_single
        if age0.dtype == np.float32
        else _cablesim.run_chunk
    )
    return simulate.Results(
        *at_width(
            length_ft,
            customers,
            customer_minutes_per_failure,
            customer_minutes_per_planned,
            outage_cost_per_failure,
            class_index,
            age0,
            shape,
            scale,
            replacement_shape,
            replacement_scale,
            cost_per_ft,
            draw_key,
            first_replication,
            n_reps,
            budget,
            cost_escalation,
            policy,
            emergency_multiplier,
            mobilization_per_segment,
            emergency_charged_to_budget,
            n_classes,
            n_years,
            threads,
        )
    )
