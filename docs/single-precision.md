# Single precision, measured

Every figure is seconds per replication and peak resident memory on a 48-core
machine, release build, one implementation per process so the memory attributes
to one of them. Every configuration reproduced the scalar reference **exactly,
in every cell of every array, at its own width** — the parity suite runs at both
precisions with no tolerance either way.

The two precisions give different answers. That is the choice, not a defect: at
12,000 segments the difference shows only in the dollar columns, at about 1e-7
relative, while the counts and which segments were funded are identical.

## Can the two languages agree in single precision?

This gated everything else, because the validation strategy is bit-for-bit
agreement and in single precision NumPy carries its own vectorised loops for
`expm1`, `log1p` and `pow` where the crate calls the system `expm1f`, `log1pf`
and `powf`. Neither is required to be correctly rounded.

**They agree.** The three Weibull forms are identical over two million samples
of the ages and uniforms the model produces, over all 160 distinct shape and
scale pairs the shipped fleet carries, and at the edges — a zero uniform, the
smallest and largest a draw can represent, age zero. Computing the hazard with
`exp` instead of `exp_m1` on one side makes 76% of it differ, so the comparison
discriminates.

The likely reason is that both sides evaluate these functions by widening to
double, computing there and rounding once. **That explanation also predicts the
timings below**, and is the single most useful thing this study found.

## What it costs and saves

### 12,000 segments — the shipped population

| | seconds, f64 | seconds, f32 | | memory, f64 | memory, f32 | |
|---|---|---|---|---|---|---|
| Kernel, 48 threads, `age_threshold` | 0.000239 | 0.000264 | **0.90x** | 34.1 MB | 18.4 MB | **46% less** |
| Kernel, 48 threads, `risk_ranked` | 0.001337 | 0.001339 | 1.00x | 48.0 MB | 30.8 MB | 36% less |
| Batched NumPy, `age_threshold` | 0.03708 | 0.03630 | 1.02x | 71.5 MB | 53.4 MB | 25% less |
| Batched NumPy, `risk_ranked` | 0.03927 | 0.03861 | 1.02x | 79.2 MB | 59.2 MB | 25% less |

**No speed at all, and slightly slower in one cell.**

### 100,000 segments

| | seconds, f64 | seconds, f32 | | memory, f64 | memory, f32 | |
|---|---|---|---|---|---|---|
| Kernel, 48 threads, `age_threshold` | 0.003014 | 0.001242 | **2.43x** | 274.5 MB | 144.8 MB | 47% less |
| Kernel, 48 threads, `risk_ranked` | 0.014016 | 0.008796 | **1.59x** | 372.0 MB | 244.1 MB | 34% less |
| Batched NumPy, `risk_ranked` | 0.42947 | 0.36766 | 1.17x | 227.5 MB | 150.0 MB | 34% less |
| Scalar reference, `risk_ranked` | 0.40756 | 0.34929 | 1.17x | 14.1 MB | 1.9 MB | 87% less |

## What that means

**The speedup is cache residency, not arithmetic.** It is absent at 12,000
segments and worth 1.6x to 2.4x at 100,000, on the same code and the same
policies. Halving the working set matters when forty-eight workers are pulling
their scratch through a shared cache and does not matter when it already fits.

**This contradicts the prediction that opened the plan**, which was drawn from
NumPy microbenchmarks: single precision was about twice as fast on elementwise
arithmetic and level on sorting, so the expectation was roughly 2x where a
policy is arithmetic-bound and nothing where it is sort-bound. The measured
split is by *population size* instead, and it helps the sort-bound policy
almost as much as the other. Those microbenchmarks timed one thread on isolated
arrays, which is the wrong shape for a kernel running forty-eight.

**The arithmetic itself is no faster, and step 1 says why.** If both widths
evaluate `powf`, `expm1` and `log1p` by widening to double, then the two agree
bit for bit *and* single precision buys nothing per operation — one mechanism
explains both results. Everything gained here is bandwidth and cache.

## Where it is worth using

**At 100,000 segments and above, on the kernel.** That is where it is 1.6x to
2.4x, and where halving 274 MB of scratch is worth something.

**Not at the shipped 12,000.** It is a wash on time, and 34 MB against 18 MB is
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
`tests/test_benchmarks.py` now checks the width directly, which is the only
thing that can.
