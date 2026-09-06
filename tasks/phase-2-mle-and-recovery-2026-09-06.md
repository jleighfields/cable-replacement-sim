# Phase 2 — censored MLE and the recovery ladder

The phase that decides whether the failure model can be fitted at all. It is
also where the length claim is confirmed or abandoned, so nothing that depends
on that claim should be built until it is settled.

Everything here fits; nothing simulates. The annual loop is the phase after.

## What it has to deliver

- `records.py` — the synthetic episode-grain failure history.
- `weibull.py` gains the censored, left-truncated log-likelihood and the fit.
- Five rungs of the recovery ladder, plus a cross-check against an independent
  implementation.
- Notebook 02, the censored-MLE walkthrough.

## The order to build it in, and why

The ladder is the deliverable, so the order is the order its rungs need. Each
step below is a commit that leaves the suite green.

### 1. `records.py`, rungs 1 and 2 only

The record generator takes its parameters as arguments rather than reading them
all from the configuration, because rungs 1 to 3 need single-technology data
with controlled lengths and the shipped `records:` block describes the full
population mix. Only rungs 4 and 5 use the configured block.

Grain is one row per **installation episode**, not per segment: a segment
installed in 1972, failed in 2006 and still in service contributes two rows,
one uncensored lifetime of 34 years and one censored at 20. Collapsing to one
row per segment discards the failure or mismeasures its age, and both bias the
fit long.

Lifetimes stay **continuous** — fractional years in a column named for years —
because the likelihood is continuous-time. Rounding makes it interval-censored
data fitted with the wrong likelihood, and rung 1 then fails for a reason that
is not a bug in the code.

Episodes that ended at or before `monitoring_start` are dropped: never
observable, and dropping them is what the truncation correction compensates
for. The comparison is inclusive so no episode is kept with zero exposure after
entry.

### 2. The likelihood, and rungs 1 and 2

Right-censored first, then left truncation as a separate term, because that is
the boundary between the two rungs and the point of separating them.

Fit on `(log k, log lambda)` so the parameters stay positive. Starting values
and bounds are a judgment call — record what was chosen and why in the module.

**Rung 1** isolates the censored likelihood. **Rung 2** adds truncation, and it
is checked at three entry ages spanning the range: the correction's error grows
with entry age, so a single young cohort passes with it missing entirely.

### 3. Covariates, and rung 3

Two covariates, `log(n)` and `log(L / L_ref)`, never one composite: they enter
the effective scale at different strengths, so one coefficient on both recovers
neither.

Rung 3 is the phase's reason for existing. The conductor coefficient must come
back at `-1/k` — derived, exact, not tunable. The length coefficient should come
back at `-beta/k`, which checks that the generator and the estimator agree
about the configured exponent rather than predicting it independently.

**If the conductor coefficient does not recover, stop.** Either the reduction
or the regression specification is wrong, and every later rung inherits it.

### 4. Technology indicators, rungs 4 and 5

Reference coding: one technology is the baseline and carries no indicator.
Intercept plus an indicator per level is rank-deficient and a solver either
fails or returns one of infinitely many answers.

Rung 4 is common shape; rung 5 puts the indicators in the ancillary term so
shape varies too. The two are separate because the generator and the estimator
must agree about whether shape varies — fitting a common-shape model to data
generated with per-technology shape recovers a compromise and biases every
scale, and it looks like a tolerance problem rather than a specification error.

Each rung generates data matching its own specification.

### 5. The independent cross-check

`lifelines.WeibullAFTFitter` fits the same data in the test suite and must agree
on point estimates to within a tenth of the fitted standard error. Point
estimates rather than intervals: comparing intervals tests the two libraries'
interval machinery instead of the likelihood.

Its parameterization is `lambda(x) = exp(beta'x)` with shape constant unless
given ancillary covariates — the conversion to `(k, lambda)` is written out in
the design and must be applied rather than recalled.

### 6. Notebook 02

Walks the interface layer by layer, as notebook 01 does. Shows the likelihood
surface, the fitted against the true survival curve, and why censoring must be
handled — the same data fitted ignoring censoring, so the bias is visible
rather than asserted.

The coverage study belongs here rather than in the suite: refitting a hundred
times to confirm roughly 95% of intervals contain the truth is a figure worth
showing and minutes too slow to gate a merge.

## How each rung is asserted

- **The truth inside the fitted 95% interval**, at a pinned seed — not a point
  estimate within a fixed epsilon, which either flakes or proves nothing.
- **Every parameter in a rung simultaneously.** Stricter than each separately,
  and safe only because the seed is pinned, which turns a coverage question
  into a fixed check.
- **Sized by power, not realism.** Recovering a shape to a useful interval
  takes on the order of hundreds of *observed* failures per cell, and heavy
  censoring means most cable never fails inside the window. Raise the sample
  until the intervals are tight, and record the number with that reasoning.

## Open questions to settle while building

1. **Does `records.n_segments` survive the calibration?** It was set before the
   technology parameters moved. Under the current scales the newest technology
   accumulates few observed failures, and rungs 4 and 5 need enough per cell.
   Re-derive it from the recovery test rather than assuming it.
2. **Does rung 5 stay in the suite or move to the notebook?** It needs the most
   data and will be the slowest. Decide once it has been run and timed.
3. **Optimizer starting values and bounds.** A judgment call; the requirement is
   that whatever is chosen is written down beside the code.

## Done when

- Every rung passes at a pinned seed, and each was watched failing against a
  deliberately wrong likelihood before being trusted.
- The cross-check agrees.
- Notebook 02 runs headless.
- The design's open item about the record table's size is answered with a
  number and the reasoning that produced it.
