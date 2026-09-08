# Phase 6 — the cost frontier, and the lost load it needs

## What this phase is

`PLAN.md` gives Phase 6 three parts: a new result field `voll`, the
reliability-index trajectory figures, and the cost frontier scatter.

**The trajectory figures already exist**, so this plan does not build them.
`notebooks/04_policy_explorer.py` summarizes `saidi` and `saifi` out of
`metrics.indices_per_replication` and draws each through `plots.trajectory`,
for every policy at one budget; `notebooks/06_production_run.py` draws the same
pair at production scale. Both are driven by the indices rather than by raw
counts, which is what that part of the phase asked for. The only work left
there is correcting the Phase 6 entry in `PLAN.md` so it stops describing built
work as pending.

That leaves two things: the quantity, and the figure that needs it.

## The quantity

`Results` carries `planned_spend` and `emergency_spend` — what the utility
pays — and nothing for what customers lost. `outage_cost_per_failure` is
already built per segment in `population.py` from
`reliability.voll_per_customer_hour` and the segment's customer counts, and it
already crosses the boundary into every implementation, because the
`risk_ranked` score reads it. It is simply never accumulated.

**Add one result field, `voll`**: the eighth and last, totalling
`outage_cost_per_failure` over the segments that failed, carrying the year's
`cost_escalation` — the same multiplier the ranking score applies to the same
column, so a dollar of lost load and a dollar of construction move together.

Three decisions that go with it, stated here because each is a place the
implementation could reasonably go the other way:

- **It is not charged to any budget.** It is value lost by customers, not
  money the utility spends, so it enters neither the planned budget nor the
  `emergency_charged_to_budget` path. Charging it would make the budget
  constraint respond to a quantity the utility never pays.
- **It stays out of `total_spend`.** That column is utility spend, and the
  reliability-against-budget figures divide by it. The frontier's y-axis gets
  its own derived column instead, `failure_cost = emergency_spend +
  voll`, which is what `PLAN.md` puts on that axis.
- **It is discounted like the other dollar streams.** It accumulates nominally
  in the loop and gets a present-value counterpart in `metrics.discount`, for
  the reason the spend columns do: a year-29 dollar is not a year-0 dollar.

### Steps

1. **`simulate.Results`** — add `voll` as the eighth field, after
   `emergency_spend`, with its docstring entry saying it is customer value
   lost rather than money spent.

2. **`simulate.run_chunk`** — accumulate at the failure step, beside
   `emergency_spend`:
   `by_class(failed, outage_cost_per_failure[failed] * escalation)`.

3. **`batched.run_chunk_numpy`** — the same total through `totals_by_class`,
   gathering then multiplying, in the order the emergency cost beside it uses,
   so the two agree bit for bit.

4. **`src/simulate.rs`** — the field on `Results`, in `zeros`, in `append`, and
   accumulated in the failure loop as
   `(outage_cost_per_failure[segment] * escalation).into_double()`. That is the
   expression the score site at the same working width already uses, which is
   what keeps the parity exact rather than approximate.

5. **`src/lib.rs`** — an eighth `as_result_array` entry, in field order, and
   the returns doc listing it.

6. **`kernel.py`** — `simulate.Results(*arrays)` is positional and needs no
   change; its docstrings do.

7. **`results.SCHEMA`** — `voll: pl.Float64`. Every returned array is
   saved, without exception, because the metrics read this file and nothing
   else.

8. **`metrics`** — sum it in `indices_per_replication` and `horizon_totals`;
   derive `failure_cost` beside `total_spend`; discount both in `discount`.

### Naming

The field is `voll`, lowercase, because every other result field and saved
column is lowercase and a Rust struct field spelled `VOLL` warns under
`non_snake_case` — so the uppercase form would need a lint suppression to sit
in the one place the mirror makes it load-bearing.

Two names then sit close together and mean different things, so each says which
it is in its own docstring: `reliability.voll_per_customer_hour` in the config
is the **rate**, dollars per customer-hour by customer type; `voll` in the
results is **accumulated dollars**, that rate carried through a segment's
customer counts and restoration time and totalled over the failures of a year.

**The per-segment input keeps the name it has**, `outage_cost_per_failure`. It
is in the kernel signature, the binding, the population generator and the
mirror table, and renaming it is a separate change with a much wider blast
radius than this phase needs. The result being `voll` while its input is
`outage_cost_per_failure` is the one seam this leaves; say so if you would
rather align both, and that becomes its own commit.

### The word "seven"

The count of result arrays is written out in eighteen places across
`kernel.py`, `run.py`, `simulate.py`, `batched.py`, `metrics.py`, `src/lib.rs`,
`src/simulate.rs`, `tests/test_parity.py`, two notebooks and five spots in
`PLAN.md`. Adding an eighth array makes every one of them wrong, and this is
the defect the repo keeps meeting: a number written in many places, where the
copies drift silently.

**Drop the numeral wherever it carries nothing** — "the per-year, per-class
arrays" says as much as "the seven per-year, per-class arrays" to any reader,
and it does not go stale. Keep an explicit list only in the two places where
the list itself is the contract and its order matters: the `Results` doc
comment in `src/simulate.rs`, and the returns section of the binding, which
tells a caller what position each array arrives at.

## The figure

`plots.cost_frontier(...)`, new: **planned spend actually incurred on x** —
the money spent, not the budget offered, so a policy that cannot spend its
allowance shows that — and **`failure_cost` on y**. One point per policy per
budget level, over the sweep `scripts/budget_sweep.py` already produces.

**Every replication is drawn faintly behind the policy means**, rather than
error bars on each axis. The two costs are correlated within a replication — a
year of many failures raises both — and a pair of error bars states each margin
while hiding exactly that correlation. So the figure takes the per-replication
horizon totals as well as the policy means, which is one argument more than
`reliability_against_budget` takes.

At 1,000 replications and eight budget levels the cloud is 40,000 points across
five policies. That is the one figure in this project whose rendering cost is
worth measuring, so measure it and record what it costs; if it is too slow to
be interactive, the fix is a decision to make with a number in hand rather than
a guess up front.

The frontier goes in `notebooks/04_policy_explorer.py`, in the budget-sweep
section that already runs the sweep and draws `reliability_against_budget` from
it, rather than in a new notebook — the data it needs is already computed
there.

## Tests

Parity gets the new array for free: `assert_identical` iterates
`Results._fields`, so every existing comparison covers it the moment all three
implementations have the field. What that does **not** cover is whether the
value is right, since three implementations agreeing on a wrong number is
exactly what a mirror cannot detect. So:

- **An analytical test** in `tests/test_simulate.py`: a population where the
  segments that fail are known, at known values of lost load, under a known
  escalation, asserting the dollar total exactly. This is worth more than a
  parity check because it can be wrong in only one way.
- **A test that the budget does not see it** — the funded set under a binding
  budget must not move when `outage_cost_per_failure` changes.
- **A test that `total_spend` does not include it**, in `tests/test_metrics.py`.
- **Column tests**: `tests/test_results.py` asserts the saved schema and
  `tests/test_metrics.py` builds rows by name; both need the new column.
- **A figure test** in `tests/test_plots.py`: the frontier draws a mean point
  per policy per level and a cloud behind it, and plots the values it was given.

**Each of these gets watched failing against a named mutation** before it is
believed: drop the escalation from the accumulation; total over every segment
rather than the failed ones; add the lost-load value into `available`; add it into
`total_spend`; write it into the wrong class cell on the Rust side.

## What else has to move

- **`PLAN.md`**: the Phase 6 entry (the trajectory figures are built), section
  7.2's column list, the binding's return description in 5.2, the mirror table
  row, and the "seven arrays" spots.
- **`notebooks/05_parity_and_bench.py` and `06_production_run.py`**: the array
  count in their prose.
- **The memory figures in `run.py`**. `UNBOUNDED_BATCH` cites ~58 MB per
  thousand replications, of which the result arrays are about 5. An eighth
  array raises the array part by roughly a seventh of that, near 1% of the
  total — inside the spread already recorded, but the claim is a measurement
  and gets rechecked rather than assumed to survive.
- **`README.md`**, if the public surface it describes changes.

No new configuration knob: the value of lost load is already parameterised, and
this phase only accumulates what the config already produces.

## One liability to state, not to fix here

`PLAN.md` section 13.2 records the value-of-lost-load figures in
`configs/base.yaml` as placeholders needing sourcing to published
interruption-cost estimates. Until now they moved a ranking; after this phase
they denominate an axis of the deliverable figure in dollars. The plumbing is
the same whatever the numbers are, so this does not block the work — but the
frontier's notebook cell must say the y-axis is placeholder-valued, or the
figure reads as a costed result it is not.

## Order of work

1. The field, through all three implementations and the binding. Rebuild with
   `uv run maturin develop --release`, then the parity suite.
2. The schema, the metrics columns, and their tests.
3. The analytical and negative tests, each watched failing.
4. The frontier figure and its test.
5. The notebook cell, and `pytest -m notebooks`.
6. The prose sweep: the numeral, `PLAN.md`, the notebooks, `README.md`.
7. `code-reviewer commit` per commit; the full pass before the pull request.

## Review

To be written when the work is done.
