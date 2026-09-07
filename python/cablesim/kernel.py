"""The Rust compute kernel, behind the reference's own call signature.

`simulate.py` is the correctness reference and this is the fast path, and the
point of this module is that nothing above them can tell which it holds. The
call takes the same twenty-two arguments under the same names and returns the
same `simulate.Results`, so `run.py`, the metrics layer and every test drive
both through one path rather than through two that could differ in how they are
driven.

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
    lifetime_uniforms: np.ndarray,
    policy_uniforms: np.ndarray,
    budget: np.ndarray,
    cost_escalation: np.ndarray,
    policy: policies.Resolved,
    emergency_multiplier: float,
    mobilization_per_segment: float,
    emergency_charged_to_budget: bool,
    n_classes: int,
    n_years: int,
) -> simulate.Results:
    """Runs one chunk of replications under one policy, in Rust.

    The arguments are `simulate.run_chunk`'s, and mean the same things. Every
    array crosses the boundary as the bytes NumPy already holds rather than as
    a copy, so the two implementations read the identical draws and there is no
    random-number stream to reconcile across the two languages.

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
        lifetime_uniforms: ``(replications, segments, n_years + 1)`` uniforms.
            Index 0 is the left-truncated draw made at the start of the run,
            and a replacement made in year ``y`` reads index ``y + 1``.
        policy_uniforms: ``(replications, segments)``, one fixed priority per
            segment, which only the random policy ranks on.
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

    Returns:
        The seven per-year, per-class arrays for this chunk.

    Raises:
        ValueError: If the policy tag names no policy, if the population is
            empty, if ``n_classes`` is 0, if a class index is past the end of
            the class axis, if an array is not C-contiguous, if a per-segment
            or per-year array is the wrong length, if the draw array is not
            the shape the horizon implies, or if a candidate scores a rank key
            that is not a number.
        TypeError: If an array's dtype is not the one the boundary reads —
            ``uint8`` for ``class_index`` and ``float64`` for the rest. The
            binding does not convert, because converting would copy and a copy
            of the draw array is the largest thing in a run.
    """
    return simulate.Results(
        *_cablesim.run_chunk(
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
            lifetime_uniforms,
            policy_uniforms,
            budget,
            cost_escalation,
            policy,
            emergency_multiplier,
            mobilization_per_segment,
            emergency_charged_to_budget,
            n_classes,
            n_years,
        )
    )
