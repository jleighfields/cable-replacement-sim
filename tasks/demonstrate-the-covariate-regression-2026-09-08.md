# Demonstrate the failure-time regression

## The gap

`PLAN.md` section 2.4 specifies a failure-time regression: two geometry
covariates, `log(n)` and `log(L / L_ref)`, never one composite; technology as
indicators in both the location and the ancillary terms so shape can vary. The
package implements all of it — `records.geometry_covariates`,
`records.technology_indicators`, `weibull.fit_regression` — and the recovery
ladder validates it, rungs 3 through 5.

**No notebook shows it.** Notebook 03 walks the likelihood, the truncation
correction, the censoring term and the confidence intervals, and every fit in
it goes through `weibull.fit_censored`, which calls the regression with an
empty design matrix. So the covariate path — the part of section 2.4 that is
actually a modelling decision rather than bookkeeping — has tests and no
demonstration.

That inverts this repository's own rule: the package implements, the notebooks
demonstrate, the reference validates. A simplify audit flagged the two record
functions as reachable only from tests, which is the same fact read as dead
code. They are not dead; they are undemonstrated.

## What this is not

**Not a replay of the recovery ladder.** The ladder fits data generated at
known parameters on a fixture built to give every technology equal exposure,
and asserts the parameters come back. That belongs in `tests/`, it is already
there, and copying it into a notebook would add a slow duplicate of a fast
test.

**The demonstration that does not exist anywhere is what the regression does on
the fleet this project ships**, where exposure is not equal because technology
follows install year. `PLAN.md` section 13.2 records that as an open question
and says the ladder "sidesteps it with a fixture that gives every technology
equal exposure, and that is a fixture rather than a claim about any fleet." The
notebook is where that claim gets made or withheld honestly.

## Feasibility, measured first

The whole fit is cheap, which was the main thing that could have made this
expensive. On the shipped configuration at 20,000 segments — 21,426 episode
rows carrying 1,426 observed failures:

| step | seconds |
|---|---|
| build the episode table | 0.03 |
| fit with geometry and technology, common shape | 0.29 |
| fit again with shape varying by technology | 0.56 |

So the demonstration costs about a second, and the notebook needs no reduced
size, no cached fit and no apology.

## What the notebook should show

A new section in `notebooks/03_weibull_fitting.py`, after the interval section
that closes it today. Four steps, each one API call, in the order the package
layers them:

1. **The episode table for the whole fleet**, rather than the single-technology
   single-conductor table the notebook uses for its pedagogy. This is the first
   place the notebook meets a population with a technology mix.
2. **The design matrix**, built by the two record functions and stacked. Show
   the columns and say what each one is: two geometry columns and one indicator
   per technology against the reference.
3. **The fit at a common shape**, and the two closed forms it can be checked
   against. Both are derived rather than tuned, which is what makes this a
   check rather than a report: the coefficient on `log(n)` is `-1/k`, and the
   one on `log(L/L_ref)` is `-beta/k`, with `beta` the configured length
   exponent of 0.5.
4. **The fit with shape varying by technology**, and its intervals.

**The interesting part is step 4, and it is where the honesty is.** The
configuration ships three technologies at shapes 6.2, 6.5 and 6.8, on vintages
1965–1985, 1986–2004 and 2005–2020. The newest is also the youngest cable, and
`PLAN.md` section 13.2 argues its shape is weakly identified because the
binding constraint is exposure time rather than sample size. So the notebook
should report each technology's interval and let the width say which
coefficients the fleet can support — including, if it turns out that way, that
one of them cannot be told from the reference.

**Do not tune the population until the answer looks better.** If a coefficient
does not come back, that is the demonstration.

## What to assert in the cells

`CLAUDE.md` allows a notebook to assert on something it has already computed
whose value is known, and this section computes several.

- **The two geometry coefficients against their closed forms**, with a stated
  tolerance. These are the assertions worth having: they are derived from the
  weakest-link reduction, so a failure means the generator and the estimator
  disagree about the model rather than that an estimate is noisy.
- **That the reference technology's shape interval covers its configured
  value.** The reference has the longest exposure and should be the one the
  fleet can identify.
- **The weak identification itself, where it is present** — assert that the
  interval is wide rather than that the estimate is right. An assertion that a
  weakly-identified parameter came back correct would be an assertion about one
  seed.

Each of these has to be watched failing. The cheapest mutations: change the
configured length exponent and confirm the `log(L/L_ref)` assertion reddens;
narrow the study window and confirm the interval-width assertion notices.

## What may have to change in the package

Possibly nothing — the four calls above are all public API, which is the test
of whether the package has a seam here. Two things to check while writing it,
because either would be a package finding rather than a notebook one:

- Whether the notebook can name the reference technology without reaching
  inside a config object in a way the package should be doing for it.
- Whether `RegressionFit` reports enough to say a coefficient is weakly
  identified, or whether the notebook has to compute interval widths itself.
  If it has to, that is a small addition to the package, not notebook logic.

## Verification

- `pytest -m notebooks`, which runs nowhere automatically, so it is run by hand
  before the pull request and named in it.
- Both notebook run modes, headless and interactive, since both must keep
  working.
- The in-cell assertions watched failing against the mutations above.
- The runtime of the new section measured and stated, so the next person
  changing notebook 03 knows what it costs.

## What this does not do

It does not answer whether the newest technology's parameters should be
reported, pooled with the previous technology, or carried from a prior — the
three ways out that `PLAN.md` section 13.2 lists. It shows which of them the
fleet forces, which is what that decision has been waiting for.
