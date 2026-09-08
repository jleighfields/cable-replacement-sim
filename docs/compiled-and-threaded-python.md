# Numba and threaded NumPy, measured against the Rust kernel

**The Numba implementation this measures is retired.** It reached the kernel and
was moved to `deprecated/compiled_numba.py` rather than kept, because reaching
it is not a reason to maintain a fourth implementation of the model. The
findings below are why the benchmark table says what it now says, and they hold
whether or not that code ships. Threaded NumPy is still in the package, as a
thread count `batched.run_chunk_numpy` accepts.

Every figure below is seconds per replication, release build, on a 48-core
machine. Every configuration reproduced the scalar reference **exactly, in every
cell of every array** — forty-two rows, no tolerance anywhere — so the times
compare implementations of one computation rather than four different ones.

Each configuration is warmed once before the clock starts, and what that first
call cost is reported separately. Compilation is real and belongs in the
argument, but folding it into a mean over three runs describes neither the first
call nor the ones after it.

## The question

The published speedup compared **one Python thread against forty-eight Rust
threads**, and could not distinguish three explanations: that Rust generates
better code, that compiled beats interpreted, or that many threads beat one.
These four implementations separate them, because each differs from another in
exactly one respect.

## 12,000 segments, 96 replications

| Implementation | `run_to_failure` | `age_threshold` | `risk_ranked` |
|---|---|---|---|
| Scalar reference | 0.00746 | 0.02033 | 0.04405 |
| Batched NumPy, 1 thread | 0.00459 | 0.02468 | 0.04479 |
| Batched NumPy, 48 threads | 0.01792 | 0.04335 | 0.03564 |
| Numba, 1 thread | 0.00273 | 0.00314 | 0.04893 |
| Rust kernel, 1 thread | 0.00202 | 0.00266 | 0.04058 |
| Numba, 48 threads | **0.00014** | **0.00011** | **0.00134** |
| Rust kernel, 48 threads | **0.00014** | 0.00019 | 0.00148 |

## 100,000 segments, 48 replications

| Implementation | `run_to_failure` | `age_threshold` | `risk_ranked` |
|---|---|---|---|
| Scalar reference | 0.04317 | 0.14780 | 0.41316 |
| Batched NumPy, 1 thread | 0.03861 | 0.23896 | 0.45470 |
| Batched NumPy, 48 threads | 0.11990 | 0.16782 | 0.17205 |
| Numba, 1 thread | 0.02328 | 0.02704 | 0.48759 |
| Rust kernel, 1 thread | 0.01699 | 0.02300 | 0.38532 |
| Numba, 48 threads | 0.00231 | **0.00258** | 0.01661 |
| Rust kernel, 48 threads | **0.00208** | 0.00295 | **0.01416** |

## What the numbers say

**Rust is worth 1.2 to 1.4 times on one thread, and the comparison stops being
decidable on forty-eight.** Against Numba running the same algorithm, the kernel
is ahead by 1.18x to 1.37x single-threaded, in all six columns. At forty-eight
threads the two trade places by policy and by size, and the spread is wide —
Numba is 1.73x ahead at 12,000 segments under `age_threshold`, the kernel 1.17x
ahead at 100,000 under `risk_ranked`. Those 12,000-segment rows are 110 to 190
microseconds per replication, which three repeats on a shared machine cannot
separate, so the honest reading is that the language stops mattering once the
work is spread, not that either side wins.

**Compiling is worth five and a half to six and a half times — but only where
the loop is scalar-shaped.** Under `age_threshold`, Numba on one thread beats the reference
it is a translation of by 6.5x at 12,000 segments and 5.5x at 100,000. That
policy makes a few hundred segments eligible in a year, and a compiled scalar
loop touches only those where the batched loop sorts all of them. This is the
case the whole exercise was aimed at, and it is the largest single-thread effect
in the study.

**Under `risk_ranked`, compiling is worth nothing at all** — Numba is *slower*
than the reference, 0.0489 against 0.0441. That policy makes every segment a
candidate, so the work is one large sort, and NumPy's C `argsort` beats the
sort Numba generates. Compiling helps where the interpreter was the cost; it
does not help where the cost was already inside a C routine.

**Threads are worth eight to thirty-six times, and are almost the whole story.**
The term is the largest in every column, and it varies more than any other, so
it is worth reading with its conditions attached rather than as one number:

| 1 thread against 48 | `run_to_failure` | `age_threshold` | `risk_ranked` |
|---|---|---|---|
| Kernel, 12,000 segments | 14.4x | 14.0x | 27.4x |
| Numba, 12,000 segments | 19.5x | 28.5x | 36.5x |
| Kernel, 100,000 segments | 8.2x | 7.8x | 27.2x |
| Numba, 100,000 segments | 10.1x | 10.5x | 29.4x |

It reaches the high twenties under `risk_ranked` at both sizes, where each
replication is long enough that spreading it pays fully, and once elsewhere —
Numba under `age_threshold` at 12,000 segments, at 28.5x. The low end is where
the fixed cost of handing work to forty-eight workers takes a visible share: the
four cells between 7.8x and 10.5x are all at 100,000 segments with only 48
replications, which is one replication per worker, so a single straggler holds
the result.

**Threading NumPy works only where the sort is large.** Forty-eight threads made the batched
loop *slower* in three of the six columns — 3.9x and 1.8x slower at 12,000
segments under `run_to_failure` and `age_threshold`, and 3.1x slower at 100,000
under `run_to_failure`. It helped in the other three, and only where the sort is
large: 1.26x and 2.6x under `risk_ranked` at the two sizes, and 1.4x under
`age_threshold` at 100,000, where a bigger population makes that policy's sort
big enough to pay for the threads.

The reason is visible in what the operations do with the interpreter lock. Timed
in isolation, `lexsort` dominates the batched loop's cost and scales 3.94x on
four threads, and `argsort`, `cumsum`, `where` and the elementwise arithmetic
release the lock as well. But between those calls sits Python-level
orchestration that holds the lock throughout, and that is what binds: threading
helps only where the sort is large enough to dominate the orchestration, and
elsewhere the per-block overhead is pure loss. Growing the population moves a
policy across that line, which is why `age_threshold` flips from 1.8x slower to
1.4x faster between the two sizes. **An
operation releasing the lock is not enough; it has to release it for long enough
to outweigh what surrounds it.**

## What this means for the project

The honest reading is that **the kernel's advantage is compilation and threads,
and Rust contributes a modest constant on top**. A speedup quoted against
single-threaded Python is mostly measuring the absence of threads.

That is not an argument for removing the Rust kernel, and this study does not
make one. What Rust still carries, none of which is a speed claim:

* **Ahead-of-time compilation.** Numba builds machine code on first call:
  **5.4 seconds** measured with its cache cleared, 2.5 for the serial form and
  2.9 more for the parallel one. Numba writes the result beside the source, so
  the second process pays 0.13 seconds instead — but the first one after any
  edit to that module, or on any machine that has not run it, pays the full
  amount, and the Shiny app's Run button is exactly that first call.
* **One wheel across Python versions.** The kernel is built `abi3-py311` and
  runs on 3.11 through 3.14 unchanged. Numba pins `llvmlite`, which pins LLVM,
  and lags new Python releases.
* **Explicit integer semantics.** Porting Philox needed the 64-by-64 product
  split into halves, there being no 128-bit integer, and every constant typed
  `uint64` — because NumPy's promotion rules turn a `uint64` combined with an
  untyped literal into a `float64`, which yields a generator that still looks
  random. Rust made both of those decisions impossible to make silently.

And what the study says Numba carries: **no FFI boundary, no `cargo` in CI, and
one language to read**, at 1.2x to 1.4x of the kernel's single-thread speed and
level with it threaded.

## What was not measured

The first-call figure the harness records beside each row is **not** a compile
time, and is not reproduced in the tables above. It is whatever that call cost
in that process, and for the Numba rows the on-disk cache had usually been
written by an earlier row already, so those figures are warm. The 5.4 seconds above is the cold number and was taken
separately.

* **One machine, three repeats per configuration.** Enough to separate effects
  of 5x; not enough to defend the differences between Numba and the kernel at
  forty-eight threads, which should be read as a tie whichever way a given cell
  falls.
* **Memory.** Only wall time was recorded here.
