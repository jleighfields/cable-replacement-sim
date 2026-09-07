# Phase 3 — the Python reference, results, and reporting

Branch `feature/phase-3-reference`, cut from `main`. This is the phase that
produces the first end-to-end result: a saved run on disk, and the
reliability-against-budget figure the project exists to draw.

Everything here is pure Python. The Rust kernel arrives in Phase 4 and is
validated *against* what this phase builds, so the reference has to be readable
before it has to be fast.

## What lands

Seven new modules, a driver script and a notebook. `PLAN.md` §5.1, What is
implemented where, is the authority on the split; this is the order they are
built in, chosen so each step's tests can run against something that already
works.

| Step | What | Why here |
|---|---|---|
| 1 | `policies.py` — eligibility, scoring, rank key, and the one function that builds the kernel's policy struct | Pure functions over arrays; testable before any loop exists |
| 2 | `simulate.py` — the annual loop and the greedy budget fill | Needs step 1; everything else consumes what it returns |
| 3 | `results.py` — the run directory, the manifest, the writer, the sweep reader | Needs a result shape to write, and Phase 4 diagnoses parity against saved runs |
| 4 | `run.py` — chunk and policy loops, draw generation, concatenation, the only writes | Needs 2 and 3 |
| 5 | `metrics.py` — the indices, the bands, the discounting, the avoided-minutes comparison | Reduces the saved frame, so it needs a frame to reduce |
| 6 | `plots.py` — the shared plotly figures | Needs the metrics it plots |
| 7 | `scripts/budget_sweep.py` and notebook 04 | The deliverable, which needs all of the above |
| 8 | The 30-year calibration re-check | Only answerable once the loop runs; it is an open question in `PLAN.md` §13.2, Still open, item 3 |

Each step is a commit leaving the suite green.

## Step 1 — `policies.py`

Eligibility and ranking for the five policies of §2.8, Replacement policies,
as functions over per-segment arrays. The eligibility table there is the
specification, including the part that is easy to skip: eligibility means *not
already replaced this year* and nothing else, because nothing is ever out of
service.

- `score(...)` per policy, returning the rank key. `risk_ranked` carries both
  terms — customer value at risk, and the emergency-minus-planned cost the
  utility avoids — because without the second, a lateral serving three
  customers is never replaced at any age.
- `rank_by_cost` divides by **planned** cost, since that is what the budget is
  charged. It is a `risk_ranked` parameter only.
- **One function maps a validated `PolicySpec` onto the struct the Rust kernel
  will receive**, per §5.2, The call. The neutral values live there and nowhere
  else — `threshold_years` of negative infinity for the three whole-population
  policies and positive infinity for `run_to_failure`, so eligibility needs no
  branch on the policy kind. Writing a default on both sides of the boundary is
  the defect only the deterministic parity test would catch.

Tests: each policy's eligible set and rank key against hand-worked arrays; that
`risk_ranked` without its second term stops replacing cheap laterals, which is
what the term is for; that `worst_first` ignores consequence; that the struct
builder produces the stated neutral values.

## Step 2 — `simulate.py`

The annual loop of §2.9, Annual simulation loop, one replication at a time.
Returns the seven `(chunk, n_years, n_classes)` arrays §5.2 names, so `run.py`
cannot tell the reference and the kernel apart.

The rules in that section that are easy to implement wrongly and that each need
their own test:

- **Failure times are simulation time from year 0**, not ages. A segment
  starting at `age0` with drawn age-at-failure `T` fails at `T - age0`.
  Storing an age instead is wrong by decades.
- **The initial draw is left-truncated** at each segment's starting age, via
  the existing `weibull.draw_remaining_life`. Getting this wrong produces a run
  that completes and curves that look plausible.
- **Failures resolve before planned work**, so a segment that failed this year
  is not also a candidate this year.
- **A replacement enters service at the start of the following year**, and is
  age 0 when year `y+1` is scored. This is what makes the near-zero-scale
  deterministic test terminate.
- **The sort key is `(score descending, segment_id ascending)`** — total, so
  ties cannot decide the answer. `age_threshold` ranks on an integer age shared
  by thousands of segments, and the greedy stops at the first candidate that
  does not fit, so which tied segment lands last changes what gets funded.
- **The greedy stops at the first candidate that does not fit**, rather than
  skipping it to fund cheaper ones below.
- **Unspent budget does not carry forward.**
- **A replacement in year `y` takes the draw at `[r, i, y+1]`**, that segment's
  cell for the following year.
- Where `emergency_charged_to_budget` is true, the year's emergency spend is
  charged **before** the planned pass is scored.

Tests, the first of which carries most of the weight:

- **The deterministic case, forced through the ordinary inputs.** A technology
  scale near zero makes every segment fail in its first year; one far past the
  horizon makes none fail. Both arrive in the same `scale` array every run
  uses, so the test exercises the shipped path rather than a branch only tests
  reach. This is where greedy-allocation bugs surface, and it is the only test
  that pins the emergency-spend ordering when `emergency_charged_to_budget` is
  true.
- Tie-break: two segments with equal scores and different ids, with a budget
  that funds exactly one.
- Every policy at zero budget equals `run_to_failure` at any budget.
- Unspent budget does not appear in the following year.
- A segment replaced in year `y` is not a candidate again until `y+1`.

## Step 3 — `results.py`

The run directory of §7.1, What a run writes: `config.yaml` dumped from the
validated model rather than copied from the input file, `manifest.json` with
the provenance a result cannot be read without, and `results.parquet` with one
row per `(policy, replication, year, class)` carrying all seven arrays.

The sweep reader of §7.3 scans a directory of runs into one lazy frame, with
the swept parameters written into the parquet rather than recovered from
directory names. **A missing or unreadable run raises** — a sweep silently
short one budget level draws a curve that looks fine and is wrong.

Tests: round trip, with the on-disk column names and dtypes asserted
separately, because a writer and reader sharing a mistake agree with each
other. Deleting a run from a sweep directory and confirming the reader raises
rather than returning a short frame. Fixtures written into `tmp_path`; no
parquet is committed.

## Step 4 — `run.py`

Build the segment table, generate each chunk's draws, call an implementation,
concatenate, write. The only module that writes anything.

The test that matters: **a run reassembled from chunks equals the same run
computed in one chunk.** That is the claim §2.11 makes about one child per
replication per stream, and it is what lets the benchmark sweep chunk size and
still say it changes no result.

Also: `config.yaml` records the driver's overrides, not the file's defaults.
That is the failure the convention exists to prevent, so it gets its own test.

## Step 5 — `metrics.py`

SAIFI, SAIDI, CAIDI and CMI **from the unplanned columns only**, with planned
customer-minutes reported as its own series. Replication mean and percentile
bands, horizon totals, customer-minutes avoided against `baseline_policy`, and
cost per customer-minute avoided both nominal and discounted.

Tests are analytical rather than comparative: a small hand-built frame whose
indices can be computed by hand. An analytical check can be wrong in only one
way.

## Step 6 — `plots.py`

The figures the notebooks and the app share, in plotly, each taking a frame and
returning a figure — none reads a file, none draws to a global. **plotly is an
optional extra**, imported at module scope so installing the package for the
kernel alone does not pull a plotting stack and importing `plots` without it
fails immediately and says why. This adds a dependency group.

Tests assert trace counts and the values attached to each trace, never pixels:
an image snapshot fails on a library upgrade and passes on a wrong number.

## Step 7 — `scripts/budget_sweep.py` and notebook 04

The default grid is zero plus seven levels spaced geometrically from an eighth
of `budget.annual` to twice it, so the region near a binding constraint is
sampled more densely than the flat region beyond it. **Zero is in the grid
because every policy at zero budget must equal `run_to_failure`**, which is a
free end-to-end check on the whole stack.

Notebook 04 is the reliability-against-budget explorer, and it walks the API
layer by layer rather than making one top-level call.

## Step 8 — the 30-year calibration re-check

`PLAN.md` §13.2, Still open, item 3 says the configuration was calibrated at
year zero and that the other 29 years are unchecked. This phase is the first
time that can be answered. Record the horizon behaviour beside the year-zero
numbers: a failure rate at year 30 several times the year-1 rate is a finding
if it is the ageing story and a miscalibration if `run_to_failure` runs away.

The answer goes in `PLAN.md`, not only in a notebook.

## Questions that were open, and how they were settled

1. **How scalar is `simulate.py`?** Settled: **unbatched over replications,
   vectorized over segments within a year.** One replication at a time, with
   NumPy for the scoring, the sort and the cumulative-cost cut. The distinction
   from the batched implementations of Phase 5 is the replication axis, which
   is what "batched" names there anyway.

   Measured on the per-year shape of the work — score every in-service segment,
   sort, cumulative-cost cut, apply — at 12,000 segments over a 30-year
   horizon:

   | Reading | Per replication | Full sweep, 50 reps | Full sweep, 1000 reps |
   |---|---|---|---|
   | Vectorized over segments | 39 ms | 1.3 min | 26 min |
   | Scalar over segments | 545 ms | 18 min | 6.1 hours |

   The sweep figures are eight budget levels times five policies. Both are
   lower bounds: the probe omits failure resolution, replacement draws and the
   per-class accumulation, so the real loop costs more, in the same ratio. A
   first probe put the gap at 7x rather than 14x by leaving scoring vectorized
   and scalarizing only the greedy fill — which flatters the scalar reading,
   because the fill breaks as soon as the budget is gone and never sees most
   segments, while scoring sees all of them every year.

   The per-replication loop still accumulates across years sequentially, so the
   reduction-order distinction §6.4, Implementation parity, relies on survives:
   exact agreement on discrete outcomes, `1e-12` relative on money and minutes.

2. **What size does the headline figure run at?** The premise of the original
   question was wrong. It asked what the continuous integration budget allows,
   and the answer is that **the notebooks no longer run in continuous
   integration at all** — that changed in its own branch while this plan was
   being written, so nothing external constrains notebook 04's size.

   What actually constrains it: `results/` and `*.parquet` are both ignored, so
   a notebook that loads a saved sweep has nothing to load on a fresh clone.
   Notebook 04 therefore computes its own small sweep — enough to draw a real
   figure and to exercise every layer — and **states in its own text the
   replication count it ran at**, since a figure whose replication count is not
   stated cannot be read.

3. **What does `scripts/budget_sweep.py` default to?** Settled: **a reduced
   default, with a flag to go full size.** The default should finish while
   someone watches it. The cost of this choice is that the default stops
   matching the shipped configuration once the Rust kernel makes full size
   cheap, so Phase 4 should revisit it rather than inheriting it silently.

4. **Does `plots.py` land here?** Settled: **yes.** The Shiny app is on the
   roadmap as the second caller, and building the figure twice is what the
   module exists to prevent. This adds plotly as an optional dependency group
   in this phase.

## Carried into Phase 4

- Revisit the sweep script's reduced default once the kernel makes full size
  affordable.
- Regenerate the headline figure at full replication count.
- The reference built here is what the kernel is validated against, so its
  greedy fill, tie-break and replacement timing are the contract Phase 4 has to
  match exactly, not approximately.

## Done when

- A run writes a directory that reads back, with the schema asserted
  separately from the round trip.
- The deterministic case passes with lifetimes forced through the ordinary
  inputs, in both directions.
- Every policy at zero budget equals `run_to_failure`.
- A chunked run equals the same run in one chunk.
- Notebook 04 draws the reliability-against-budget figure, and states the size
  it was run at.
- The 30-year calibration behaviour is recorded in `PLAN.md` beside the
  year-zero numbers.
