# Single precision, measured

Every figure is seconds per replication and peak resident memory on a 48-core
machine, release build, one implementation per process so the memory attributes
to one of them. **Memory is `ru_maxrss` for the whole process**, which includes
about 126 MB of interpreter and imported libraries before any array exists —
the same in every row, so the difference between two rows is the model's and
the ratio between them is not. Every configuration reproduced the scalar reference **exactly,
in every cell of every array, at its own width** — the parity suite runs at both
precisions with no tolerance either way.

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
single-precision narrowing reddens fifty of its cases.

**They agree because there is one implementation and not two.** NumPy's
single-precision `expm1`, `log1p` and `pow` are bit-identical to this platform's
`expm1f`, `log1pf` and `powf` — 20,000 of 20,000 each — which are the routines
the crate calls. They are *not* a double computation rounded once: that
explanation matches NumPy for 90% of `expm1` inputs and 93% of `log1p`, so it is
ruled out rather than merely unnecessary.

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
| Kernel, 48 threads, `age_threshold` | 0.000239 | 0.000264 | **0.90x** | 176.6 | 160.9 | 15.7 |
| Kernel, 48 threads, `risk_ranked` | 0.001337 | 0.001339 | 1.00x | 188.8 | 172.8 | 16.0 |
| Batched NumPy, `age_threshold` | 0.03708 | 0.03630 | 1.02x | 214.1 | 196.2 | 17.9 |
| Batched NumPy, `risk_ranked` | 0.03927 | 0.03861 | 1.02x | 221.8 | 201.2 | 20.6 |

**No speed at all, and slightly slower in one cell.**

### 100,000 segments

| | seconds, f64 | seconds, f32 | | peak MB, f64 | peak MB, f32 | saved |
|---|---|---|---|---|---|---|
| Kernel, 48 threads, `age_threshold` | 0.003014 | 0.001242 | **2.43x** | 331.5 | 264.6 | 66.9 |
| Kernel, 48 threads, `risk_ranked` | 0.014016 | 0.008796 | **1.59x** | 383.8 | 315.5 | 68.3 |
| Batched NumPy, `risk_ranked` | 0.42947 | 0.36766 | 1.17x | 507.5 | 403.7 | 103.8 |
| Scalar reference, `risk_ranked` | 0.40756 | 0.34929 | 1.17x | 259.7 | 228.5 | 31.2 |

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

**At 100,000 segments and above, on the kernel.** That is where it is 1.6x to
2.4x, and where the 67 MB it saves starts to be worth having.

**Not at the shipped 12,000.** It is a wash on time, and the 16 MB it saves is
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
