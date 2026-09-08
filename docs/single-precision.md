# Single precision, measured

Every figure is seconds per replication and peak resident memory on a 48-core
machine, release build, one implementation per process so the memory attributes
to one of them. **Memory is `ru_maxrss` for the whole process**, which includes
about 126 MB of interpreter and imported libraries before any array exists —
the same in every row, so the difference between two rows is the model's and
the ratio between them is not.

The timings come from `scripts/run_benchmarks.py`, which takes `--precision`,
`--segments` and `--policy` and records all three in its provenance. The memory
columns come from `scripts/measure_memory.py`, which measures one
implementation per process because a peak is a property of the process and a
program running three of them cannot say which needed it.

**Seconds are per replication and memory is not**, so the two columns are read
differently. A peak grows with how many replications are in flight — the same
kernel row is 383 MB at 24 and 567 MB at 48 — so every memory figure here is at
**24 replications**, and the count is stated because two figures taken at
different counts compare nothing.

**The policy moves it too**, by about 50 MB at 100,000 segments — three
quarters of the saving this column exists to show — so every row carries its own
measurement rather than borrowing the row above it.

Every configuration reproduced the scalar reference **exactly, in every cell of
every array, at its own width** — the parity suite runs at both precisions with
no tolerance either way.

The two precisions give different answers. That is the choice, not a defect: at
12,000 segments the difference shows only in the dollar columns, at about 1e-7
relative, while the counts and which segments were funded are identical.

## Can the two languages agree in single precision?

This gated everything else, because the validation strategy is bit-for-bit
agreement and nothing requires a single-precision `expm1`, `log1p` or `pow` to
be correctly rounded. Two implementations of them may legitimately differ in the
last bit, and the whole approach depends on their not doing so.

**They agree.** Measured before anything was built on it, through a temporary
binding: the three Weibull forms were identical over two million samples of the
ages and uniforms the model produces, over all 160 distinct shape and scale
pairs the shipped fleet carries, and at the edges — a zero uniform, the smallest
and largest a draw can represent, age zero. Computing the hazard with `exp`
instead of `exp_m1` on one side made 76% of it differ, so the comparison
discriminated.

That binding is gone, because the question it answered is now answered
continuously and by something stronger: **the parity suite runs at both widths**,
comparing NumPy against the crate through the whole annual loop rather than
three functions in isolation, at no tolerance. Perturbing the crate's
single-precision narrowing — perturbing `Real::from_double` for `f32` by one
part in a million — reddens fifty-two of its cases.

**They agree because there is one implementation and not two.** NumPy's
single-precision `expm1`, `log1p` and `pow` are bit-identical to this platform's
`expm1f`, `log1pf` and `powf` — 20,000 of 20,000 each — which are the routines
the crate calls. They are *not* a double computation rounded once: that
explanation matches NumPy for 17,936 of 20,000 `expm1` inputs drawn uniformly on
[-3, 3], 18,522 of 20,000 `log1p` inputs on (-0.99, -0.01], and 19,985 of 20,000
`pow` inputs on [0.1, 50) raised to 6.5, so it is ruled out rather than merely
unnecessary. The margin is thinnest for `pow` — fifteen inputs in twenty
thousand — which is what its negative assertion turns on.

**This is a property of the platform's libm, not of the two languages**, and it
is the assumption the single-precision mode rests on. A build against a
different libm, or a NumPy that grew its own vectorised loops for these three,
could break the agreement without anything in this repo changing.
`tests/test_weibull.py::test_numpy_and_the_system_library_agree_in_single_precision`
is what would notice.

## What it costs and saves

### 12,000 segments — the shipped population

| | seconds, f64 | seconds, f32 | | peak MB, f64 | peak MB, f32 | saved |
|---|---|---|---|---|---|---|
| Kernel, 48 threads, `age_threshold` | 0.000239 | 0.000264 | **0.90x** | 162.5 | 154.3 | 8.2 |
| Kernel, 48 threads, `risk_ranked` | 0.001337 | 0.001339 | 1.00x | 168.7 | 159.3 | 9.4 |
| Batched NumPy, `age_threshold` | 0.03708 | 0.03630 | 1.02x | 179.8 | 171.5 | 8.3 |
| Batched NumPy, `risk_ranked` | 0.03927 | 0.03861 | 1.02x | 184.3 | 173.9 | 10.4 |

**No speed at all, and slightly slower in one cell.** The saved column here is
8 to 10 MB against a run-to-run spread of about 3, so read it as "under ten"
rather than to the tenth; the 100,000-segment figures below reproduce within
half a percent.

### 100,000 segments

| | seconds, f64 | seconds, f32 | | peak MB, f64 | peak MB, f32 | saved |
|---|---|---|---|---|---|---|
| Kernel, 48 threads, `age_threshold` | 0.003014 | 0.001242 | **2.43x** | 331.4 | 262.7 | 68.7 |
| Kernel, 48 threads, `risk_ranked` | 0.014016 | 0.008796 | **1.59x** | 383.2 | 316.7 | 66.5 |
| Batched NumPy, `risk_ranked` | 0.42947 | 0.36766 | 1.17x | 509.5 | 405.4 | 104.1 |
| Scalar reference, `risk_ranked` | 0.40756 | 0.34929 | 1.17x | 258.5 | 228.8 | 29.7 |

## What that means

**The speedup is cache residency, not arithmetic.** It is absent at 12,000
segments and worth 1.6x to 2.4x at 100,000, on the same code and the same
policies. Halving the working set matters when forty-eight workers are pulling
their scratch through a shared cache and does not matter when it already fits.

**This contradicts the prediction this study set out to test**, which was
drawn from NumPy microbenchmarks: single precision was about twice as fast on
elementwise arithmetic and level on sorting, so the expectation was roughly 2x
where a policy is arithmetic-bound and nothing where it is sort-bound. The
measured split is by *population size* instead, and it helps the sort-bound
policy almost as much as the other. Those microbenchmarks timed one thread on
isolated arrays, which is the wrong shape for a kernel running forty-eight.

**The arithmetic itself is no faster**, and the reason is next to the one above
rather than the same as it. These three functions are scalar `libm` calls at
both widths, and a scalar call into `libm` does not get cheaper for being
narrower — the vectorised speedup that single precision gives NumPy's own
elementwise loops has no counterpart in a per-segment loop calling `powf`. So
everything gained here is bandwidth and cache.

## Where it is worth using

**At 100,000 segments and above, on the kernel.** That is where the speedup in
the table above appears at all, and where the 67 MB it saves at 24 replications
starts to be worth having — a figure that itself grows with the replication
count.

**Not at the shipped 12,000.** It is a wash on time, and the 9 MB it saves is
not a constraint on a machine with 251 GB.

## Two limits worth knowing

**Single precision has a narrower dynamic range, and the model can reach it.**
The hazard is a difference of two `(age / scale) ** shape` terms. At the
horizon's oldest age and the fleet's largest shape that is `(120 / scale) **
6.5`, which passes single precision's 3.4e38 ceiling once the scale falls below
about 1.4e-4. Both terms become infinite, their difference is a NaN, and the
ranking refuses it — where double precision, with room to 1.8e308, carries the
same input without noticing. No population this project generates comes near
it: the shipped scales are decades, where that term is around a thousand. A test
fixture did, and had to be changed.

**Parity between implementations cannot check the width.** The precision
travels as the dtype of the arrays, so an argument left at double widens
whatever it touches — and every implementation widens the same way and goes on
agreeing in every cell. This happened twice while building this: once in the
per-year cost series, and once in the parity fixtures themselves, where
eighty-eight cases were labelled `f32` and ran `f64` twice while passing.
Three places now check the width directly, which is the only thing that can:
`tests/test_benchmarks.py` over the arguments the benchmark harness builds, the
parity fixture in `tests/conftest.py` over the ones it builds itself, and the
boundary builder in `tests/test_parity.py` over the ones it assembles, which
neither of the other two reaches.
