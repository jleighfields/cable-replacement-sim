# Phase 2 — censored MLE and the recovery ladder

The phase that decides whether the failure model can be fitted at all. It is
also where the length claim is confirmed or abandoned, so nothing that depends
on that claim should be built until it is settled.

Everything here fits; nothing simulates. The annual loop is the phase after.

## What it has to deliver

- `records.py` — the synthetic failure history, one row per installation episode.
- `weibull.py` gains the censored, left-truncated log-likelihood and the fit.
- Five rungs of the recovery ladder, plus a cross-check against an independent
  implementation.
- Notebook 02, the censored-MLE walkthrough.

## The order to build it in, and why

The ladder is the deliverable, so the order is the order its rungs need. Each
step below is a commit that leaves the suite green.

### 1. `records.py`, rungs 1 and 2 only — done

The record generator takes its parameters as arguments rather than reading them
all from the configuration, because rungs 1 to 3 need single-technology data
with controlled lengths and the shipped `records:` block describes the full
population mix. Only rungs 4 and 5 use the configured block.

One row per **installation episode**, not per segment: a segment
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

### 2. The likelihood, and rungs 1 and 2 — done

Right-censored first, then left truncation as a separate term, because that is
the boundary between the two rungs and the point of separating them.

Fit on `(log k, log lambda)` so the parameters stay positive. Starting values
and bounds are a judgment call — record what was chosen and why in the module.

**Rung 1** isolates the censored likelihood. **Rung 2** adds truncation, and it
is checked at three entry ages spanning the range: the correction's error grows
with entry age, so a single young cohort passes with it missing entirely.

### 3. Covariates, and rung 3 — done

Two covariates, `log(n)` and `log(L / L_ref)`, never one composite: they enter
the effective scale at different strengths, so one coefficient on both recovers
neither.

Rung 3 is the phase's reason for existing. The conductor coefficient must come
back at `-1/k` — derived, exact, not tunable. The length coefficient should come
back at `-beta/k`, which checks that the generator and the estimator agree
about the configured exponent rather than predicting it independently.

**If the conductor coefficient does not recover, stop.** Either the reduction
or the regression specification is wrong, and every later rung inherits it.

### 4. Technology indicators, rungs 4 and 5 — done

Reference coding: one technology is the baseline and carries no indicator.
Intercept plus an indicator per level is rank-deficient and a solver either
fails or returns one of infinitely many answers.

Rung 4 is common shape; rung 5 puts the indicators in the ancillary term so
shape varies too. The two are separate because the generator and the estimator
must agree about whether shape varies — fitting a common-shape model to data
generated with per-technology shape recovers a compromise and biases every
scale, and it looks like a tolerance problem rather than a specification error.

Each rung generates data matching its own specification.

### 5. The independent cross-check — done

`lifelines.WeibullAFTFitter` fits the same data in the test suite and must agree
on point estimates to within a tenth of the fitted standard error. Point
estimates rather than intervals: comparing intervals tests the two libraries'
interval machinery instead of the likelihood.

Its parameterization is `lambda(x) = exp(beta'x)` with shape constant unless
given ancillary covariates — the conversion to `(k, lambda)` is written out in
the design and must be applied rather than recalled.

### 6. Notebook 02 — done

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

## Open questions, as settled

1. **Does `records.n_segments` survive the calibration?** Yes at 20,000, and
   raising it would not help what it looks like it should help. Measured on the
   shipped configuration, the share of each technology's rows that fail inside
   the window is 62.3% for the oldest, 4.6% for the middle one and 0.03% for
   the newest, so reaching 200 observed failures needs about 3,700 segments,
   12,800, and 960,000 respectively. At the configured 20,000 the first two
   draw roughly 1,100 and 335 failures, comfortably above the bar; the newest
   draws two.

   That last figure is not a sample-size problem and no sample size fixes it.
   The newest technology is installed from 2005 and so is at most 21 years old
   at a 2026 study end, where its own parameters put failure at 0.05% for the
   very oldest cohort and less for the rest. The record window, not the record
   count, is what starves it. The consequence is a fit that must not be asked
   for the newest technology's parameters at all, which is why coding an
   indicator for a technology with no observed failure now raises.

   The gap that leaves: two failures is not zero, so the guard stays quiet
   while the estimate is still meaningless. No threshold separates few from
   none, so the failure count per technology is documented as the thing to
   check rather than guessed at in code.

2. **Does rung 5 stay in the suite or move to the notebook?** It stays. The
   whole suite runs in about 5 seconds and the rung is nowhere near the cost
   that would justify moving it.

3. **Optimizer starting values and bounds.** Both fits start at an exponential
   — shape one, scale the mean age at the end of observation, coefficients zero
   — with no bounds, the parameters being fitted as logs. This is written
   beside the code and, more usefully, tested: twenty starts spanning two
   orders of magnitude in each parameter reach the same optimum to six
   decimals, under a derivative-free search so a correct gradient cannot be
   what rescues it.

## Done when

- Every rung passes at a pinned seed, and each was watched failing against a
  deliberately wrong likelihood before being trusted. **Done.**
- The cross-check agrees. **Done** — `lifelines` on both untruncated and
  truncated data. R's `survival::survreg` was run against the same table during
  development and agreed to ten decimals on every parameter and on the
  log-likelihood itself; it is not carried in the repository, since it cannot
  express delayed entry and would have run nowhere but one machine.
- Notebook 02 runs headless. **Done**, in about 10 seconds.
- The design's open item about the record table's size is answered with a
  number and the reasoning that produced it. **Done**, above.

## Review

**What the phase was for.** Deciding whether the failure model can be fitted at
all. It can, and the length claim survives: both geometry coefficients come
back inside their intervals.

**What went differently from the plan.**

- The plan treated the truncation term as a small correction kept on principle.
  It is worth 0.064 of shape here, and the notebook now demonstrates that
  rather than asserting it, by fitting the complete record the correction is
  standing in for and showing the corrected fit lands on that answer with an
  interval covering zero. The rows responsible are 0.05% of the episodes and
  0.84% of the failures, and they are the young ones.
- Two defects surfaced that the plan did not anticipate, both found by asking
  whether a term earned its place rather than by a test failing. A technology
  whose rows are all censored has no estimable scale and the fit reported
  wherever the solver stopped; and the objective was summed over episodes while
  the optimizer stops on an absolute gradient tolerance, so larger samples
  quietly demanded more precision than the line search could deliver and the
  fit raised while standing on the right parameters.
- The independent cross-check was worth more than expected. R and `lifelines`
  agreeing to ten decimals on the log-likelihood itself, not merely on the
  parameters, is a stronger statement than the plan asked for: the two are the
  same function rather than two fits that landed nearby.

**What to carry into the next phase.**

- Interval widths rest on BFGS's secant approximation to the Hessian and sit
  about 1% off a numerically exact observed information. Coverage is fine —
  96% and 94% at 200 draws — but a fit that needs exact standard errors should
  compute the Hessian rather than take the optimizer's.
- The newest technology's parameters are not recoverable from this record
  window. Anything downstream that wants them must take them from the
  configuration rather than from a fit.
