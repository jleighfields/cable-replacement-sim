# Retired implementations

Nothing here is part of the package. Nothing imports it, no test runs it, and
it is excluded from linting. It is kept so that the measurements which retired
it can be reproduced, because a finding whose evidence has been deleted is an
assertion.

| What | Retired because |
|---|---|
| `batched_polars.py` | The annual loop over a polars frame, in Python. Competitive on one policy and slower on the rest, and beaten 20–80× by the Rust kernel. |
| `batched.rs` | The same loop in Rust. Answered its question — there is no interop penalty to recover — and charged 2m22s of every Rust edit to keep asking it. |
| `compiled_numba.py` | The scalar reference's algorithm compiled by Numba. Answered its question and reached the kernel; retired because reaching it is not a reason to maintain a fourth implementation of the model, and numba pulls LLVM into every install to do it. |

## What the compiled implementation established

Its measurement is in
[docs/compiled-and-threaded-python.md](../docs/compiled-and-threaded-python.md),
and it is the reason the benchmark table no longer claims what it used to. The
published speedup compared **one Python thread against forty-eight Rust
threads**, because every Python implementation refused a larger count. This one
did not, and separating the three effects gave: threads worth 25–35x and
dominating everything, compiling worth 5–6x where a policy makes few segments
eligible, and the language worth 1.2–1.4x on one thread and about nothing on
forty-eight.

It reproduced the reference exactly in every cell, under all five policies and
at every thread count, so those figures compare one computation rather than
several.

**Reproducing it** needs `uv add --optional compiled "numba>=0.60"`, the file
moved back to `python/cablesim/compiled.py`, and `"numba"` restored to
`run.RUNNABLE`, `run.CONCURRENT` and `results.IMPLEMENTATIONS`. The parity tests
pick it up from the registry with no further change, which is what they are
written that way for.

## Why a column store is the wrong shape for this simulation

This is the part worth carrying forward, because it is a property of the tool
rather than of these two programs, and it will be true of the next frame engine
someone proposes.

**A simulation evolves state. A column store cannot evolve anything in place.**
Arrow buffers are immutable, so every write allocates a new full-length column
and copies it. The array form writes into the buffer it already has:

```python
age += 1.0
age[replaced] = 0.0        # touches what it touches
```

The frame form must produce a new `age` column of all 600,000 values to change
about 3% of them. Per year the loop rewrites four such columns at 4.8 MB each;
over a 30-year horizon that is ~576 MB of copying to change a few percent.
Measured, the state rebuild costs 0.83 ms a year against the array form's 0.20.

**`when/then/otherwise` does not avoid it either**, which is the trap: it reads
as a conditional update and is actually row-wise *selection*. polars evaluates
both branches for every row and then picks, so

```python
pl.when("replaced").then(draw_lifetime(...)).otherwise(pl.col("failure_time"))
```

computes a Weibull inverse for all 600,000 rows to use 18,000 of them —
measured at 0.377 s against 0.032 s over the loop. Filtering first is the only
way to touch fewer rows, and then the write-back allocates a full column anyway.

**The transactions are small and there are many of them.** A grouped
aggregation costs about 1.3 ms of engine dispatch however few rows it covers,
and this loop wants two per year over a few thousand contributing rows. Frame
engines are built to make one large pass efficient; a simulation is thirty
small ones that depend on each other in sequence, so the fixed cost is paid
sixty times and there is nothing to amortise it over.

**And the year axis cannot be flattened away.** Year `y+1`'s failure times
depend on which segments year `y` replaced, so the loop cannot become a single
windowed pass over a `(replication, segment, year)` frame — the shape at which
a frame engine would be at its best.

## Where a long frame genuinely won, which is worth knowing

It is not all one way, and the advantage is not obvious:

**It can filter to the eligible candidates before scoring them.** Candidate
sets differ between replications, so the rectangular NumPy form cannot compact
them without leaving a ragged array — it therefore scores every segment of
every replication. Under `age_threshold`, where 2% of segments are eligible,
the frame form is **2.1× faster** than the array form for exactly this reason.

**Its sort is faster.** The ranking sort has to be full-width for the same
ragged-candidate reason, and polars does it in 5.9 ms a year against
`numpy.lexsort`'s 9.1.

Both were swamped. The Rust kernel runs replications in parallel and is 20–80×
faster than either Python form, which is the comparison that decides how this
project is built.

## The numbers

12,000 segments, 30 years, 50 replications, release build, 48 cores,
polars 1.44.1. Seconds per replication; every row reproduced the reference
exactly.

| Policy | Batched NumPy | polars (Python) | polars (Rust) | Kernel ×48 |
|---|---|---|---|---|
| `risk_ranked` | 0.0453 | 0.0475 | 0.0601 | **0.0023** |
| `age_threshold` | 0.0253 | **0.0119** | 0.0407 | **0.00031** |
| `run_to_failure` | 0.0054 | 0.0075 | 0.0152 | **0.00025** |

**The Rust frame implementation was slower than the Python one**, which is the
answer to the question it was built to ask: there is no cost of driving polars
from Python worth recovering. Its own overhead — reading a column out copies
it, and eleven columns a year over thirty years is hundreds of megabytes the
Python side gets as a borrow — exceeded whatever it saved.

## What it cost to keep, which is what actually ended it

The Python half cost nothing to keep — it is a module nobody imports. The Rust
half pulled the polars crate into the compute crate, and that was the price:

| | Before polars | With polars |
|---|---|---|
| Cold release build | **9 s** | **297 s** (default features), **191 s** (only `lazy` and `cum_agg`) |
| Rebuild after any Rust edit | **10 s** | **142 s** |
| Continuous integration, whole job | **1m 06s** | **17m 28s** |

**The rebuild figure is the one that mattered.** A cold build is paid on a
cache miss; 142 seconds was paid on *every* edit to any Rust file, and in the
session that produced these measurements it was the single largest cost by a
wide margin — more than the simulation runs, the benchmarks and the test suite
together. Removing the dependency took it back to 10 seconds.

The continuous-integration figure is worse than it looks. Nine of those
seventeen minutes were `uv sync` building the extension in debug, which the
next step immediately rebuilt in release over the top; that waste was invisible
while the crate built in seconds and is fixed separately. But the job still ran
with 2m 32s of headroom under its 20-minute timeout, and a required check that
times out under branch protection leaves a pull request unable to merge *and*
unable to fail.

The dependency itself is 250-odd crates and about 3.3 GB of peak memory to
compile, and the final link-time optimisation pass is single-threaded, so
faster hardware does not recover it.

## What is not deprecated

**polars remains the package's frame library** for the results table, the
metrics, the sweep reader and the figures. That is work a column store is good
at: one pass, read-mostly, no state to evolve. What is retired is using it to
*run the simulation*.
