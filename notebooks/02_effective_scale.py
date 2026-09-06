import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # 02 — Effective scale

    A three-phase segment is out when any one of its conductors fails, and a
    long segment has more of itself to fail than a short one. Both shorten the
    life of the *segment* relative to the cable it is made of, and the
    simulation needs one `(shape, scale)` pair per segment rather than a
    lifetime for every conductor.

    This notebook derives the reduction that does it, and checks each step
    against sampled draws rather than asserting it. The result is what the
    whole three-phase treatment rests on, and it is also the claim the
    recovery ladder later tests by fitting — so it is worth establishing
    independently first, or the fit becomes the only evidence for the thing
    the fit is supposed to check.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from cablesim import config, random_draws, weibull
    from scipy import stats

    settings = config.load_config()
    return config, np, random_draws, settings, stats, weibull


@app.cell
def _(mo, settings):
    shape_slider = mo.ui.slider(
        1.0,
        9.0,
        step=0.1,
        value=settings.population.technologies[1].weibull.shape,
        label="shape k",
    )
    scale_slider = mo.ui.slider(
        20.0, 120.0, step=1.0,
        value=settings.population.technologies[1].weibull.scale, label="scale λ",
    )
    conductors = mo.ui.slider(1, 3, step=1, value=3, label="conductors n")
    length = mo.ui.slider(100, 2500, step=50, value=1200, label="length ft")
    mo.hstack([shape_slider, scale_slider, conductors, length])
    return conductors, length, scale_slider, shape_slider


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 10 · The minimum of n conductors

    A segment with $n$ conductors survives to $t$ only if every conductor does.
    For independent conductors sharing a shape, that product collapses:

    $$
    S_{\text{seg}}(t)
      = \Pr\!\big[\min(T_1,\dots,T_n) > t\big]
      = \prod_{i=1}^{n} S(t)
      = \Big[e^{-(t/\lambda)^k}\Big]^{n}
      = e^{-n\,(t/\lambda)^k}
    $$

    That is still a Weibull, and solving for the scale that produces it is one
    line. Set the exponent equal to the standard form and cancel $t^k$:

    $$
    n\left(\frac{t}{\lambda}\right)^{k}
      = \left(\frac{t}{\lambda_{\text{eff}}}\right)^{k}
    \;\Longrightarrow\;
    \frac{n}{\lambda^{k}} = \frac{1}{\lambda_{\text{eff}}^{k}}
    \;\Longrightarrow\;
    \boxed{\;\lambda_{\text{eff}} = \lambda\, n^{-1/k}\;}
    $$

    **The shape never moved.** That is the part worth pausing on, and the next
    section shows why it had nowhere to go.
    """
    )
    return


@app.cell
def _(conductors, np, scale_slider, shape_slider, stats, weibull):
    k = shape_slider.value
    lam = scale_slider.value
    n = conductors.value

    reduced_conductors = weibull.effective_scale(
        np.array(k), np.array(lam), np.array(n), np.array(500.0), 500.0, 0.0
    )
    closed_form = lam * n ** (-1.0 / k)

    assert np.isclose(float(reduced_conductors), closed_form)
    f"λ = {lam:.1f} over {n} conductors gives λ_eff = {float(reduced_conductors):.2f}"
    return closed_form, k, lam, n, reduced_conductors


@app.cell
def _(k, lam, n, np, random_draws, reduced_conductors, stats, weibull):
    # Sampled minima of n conductor lifetimes against the reduced Weibull.
    _source = random_draws.spawn_sources(3).population
    _draws = random_draws.uniforms(_source, n * 200_000).reshape(n, 200_000)
    minima = weibull.draw_lifetime(_draws, k, lam).min(axis=0)

    _test = stats.kstest(
        minima, "weibull_min", args=(k, 0.0, float(reduced_conductors))
    )
    assert _test.pvalue > 0.001, "sampled minima are not Weibull at the reduced scale"
    (
        f"KS against Weibull(k={k:.1f}, "
        f"λ_eff={float(reduced_conductors):.2f}): p = {_test.pvalue:.3f}"
    )
    return (minima,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 20 · Why the shape survives

    Take logs twice. For any Weibull,

    $$
    \log\!\big[-\log S(t)\big] = k\log t - k\log\lambda
    $$

    — a straight line in $\log t$ whose **slope is the shape** and whose
    intercept carries the scale. Raising the survivor to the power $n$ adds a
    constant to the left-hand side and touches nothing else:

    $$
    \log\!\big[-\log S(t)^{n}\big] = \log n + k\log t - k\log\lambda
    $$

    So on these axes the reduction is a **vertical shift of a line whose slope
    does not change**. The constant $n$ has nowhere to go except the intercept,
    and the intercept is the scale.

    This is not a property of Weibulls in general — it is a property of the
    survivor being $e^{-(\cdot)^k}$, where $t$ and $\lambda$ appear only as a
    ratio raised to the same power, so a constant multiplying the cumulative
    hazard can always be absorbed into $\lambda$. It is also exactly what lets
    the conductor and length terms compose below: two shifts of the same line
    are one shift.
    """
    )
    return


@app.cell
def _(k, lam, minima, n, np, reduced_conductors):
    import matplotlib.pyplot as plt

    _t = np.linspace(1.0, 3.0 * lam, 400)
    _single = np.exp(-((_t / lam) ** k))
    _segment = np.exp(-((_t / float(reduced_conductors)) ** k))

    _figure, _axes = plt.subplots(1, 2, figsize=(11, 3.6))

    _axes[0].plot(_t, _single, label=f"one conductor, λ={lam:.0f}")
    _axes[0].plot(
        _t, _segment, label=f"{n} in series, λ_eff={float(reduced_conductors):.1f}"
    )
    _sorted = np.sort(minima)
    _axes[0].plot(
        _sorted,
        1.0 - np.arange(len(_sorted)) / len(_sorted),
        "k--", lw=1, label="sampled minima",
    )
    _axes[0].set(xlabel="age (years)", ylabel="survival", title="survivor functions")
    _axes[0].legend(fontsize=8)

    # Weibull axes: slope is the shape, so parallel lines mean shape is intact.
    _mask = (_single > 1e-6) & (_single < 1 - 1e-6)
    _logt = np.log(_t[_mask])
    _axes[1].plot(_logt, np.log(-np.log(_single[_mask])), label="one conductor")
    _axes[1].plot(_logt, np.log(-np.log(_segment[_mask])), label=f"{n} in series")
    _axes[1].set(
        xlabel="log age", ylabel="log(-log S)",
        title=f"Weibull axes — slope is k={k:.1f} for both",
    )
    _axes[1].legend(fontsize=8)
    _figure.tight_layout()
    _figure
    return (plt,)


@app.cell
def _(k, lam, n, np, reduced_conductors):
    # The two lines are parallel to floating-point tolerance, which is the
    # claim "shape is unchanged" made numerically.
    _t = np.array([10.0, 40.0])
    def _slope(scale):
        _y = np.log(-np.log(np.exp(-((_t / scale) ** k))))
        return (_y[1] - _y[0]) / (np.log(_t[1]) - np.log(_t[0]))

    assert np.isclose(_slope(lam), k)
    assert np.isclose(_slope(float(reduced_conductors)), k)
    (
        f"slope of both lines: {_slope(lam):.6f} and "
        f"{_slope(float(reduced_conductors)):.6f}, against k = {k}"
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 30 · Length, sub-linearly

    A segment also fails when any point *along* it fails. If failure sites
    arrived as a spatial process at a rate proportional to length, a segment of
    length $L$ would behave as $L/L_{\text{ref}}$ unit pieces in series and the
    same algebra would apply with $L/L_{\text{ref}}$ in place of $n$.

    That is too strong. A large share of underground faults occur at splices,
    terminations and elbows, which scale with the count of accessories rather
    than with feet of run, so length enters raised to an exponent $\beta$
    between 0 and 1:

    $$
    S_{\text{seg}}(t) = e^{-(L/L_{\text{ref}})^{\beta}\,(t/\lambda)^k}
    \qquad\Longrightarrow\qquad
    \lambda_{\text{eff}} = \lambda \left(\frac{L}{L_{\text{ref}}}\right)^{-\beta/k}
    $$

    $\beta = 1$ is the spatial-Poisson case; $\beta = 0$ makes failure purely
    per-segment. It is also what separates the two three-phase classes from
    each other, since both carry three conductors and would otherwise be
    identical.
    """
    )
    return


@app.cell
def _(k, lam, length, np, settings, weibull):
    beta = settings.population.length_exponent
    reference = settings.population.length_ref_ft

    reduced_length = weibull.effective_scale(
        np.array(k), np.array(lam), np.array(1), np.array(float(length.value)),
        reference, beta,
    )
    assert np.isclose(
        float(reduced_length), lam * (length.value / reference) ** (-beta / k)
    )
    (
        f"β = {beta}: {length.value} ft against a {reference:.0f} ft reference "
        f"gives λ_eff = {float(reduced_length):.2f}"
    )
    return beta, reduced_length, reference


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 40 · The two together

    Both terms are constants multiplying the cumulative hazard, and by the
    argument in step 20 each is a vertical shift on Weibull axes. Two shifts of
    one line are one shift, so they compose without interacting:

    $$
    \lambda_{\text{eff}}
      = \lambda\left(n \left(\frac{L}{L_{\text{ref}}}\right)^{\beta}\right)^{-1/k}
    $$

    This is the pair the population generator stores and the simulation reads.
    A conductor count and a length go in; one shape and one scale come out, and
    nothing downstream needs to know a segment has more than one conductor.
    """
    )
    return


@app.cell
def _(beta, conductors, k, lam, length, np, reference, weibull):
    combined = weibull.effective_scale(
        np.array(k), np.array(lam), np.array(conductors.value),
        np.array(float(length.value)), reference, beta,
    )
    _stepwise = (
        lam
        * conductors.value ** (-1.0 / k)
        * (length.value / reference) ** (-beta / k)
    )

    # Applying the two reductions in turn is the same as applying the composite.
    assert np.isclose(float(combined), _stepwise)
    (
        f"n={conductors.value}, L={length.value} ft → λ_eff = {float(combined):.2f}, "
        f"median life {float(combined) * np.log(2) ** (1 / k):.1f} years"
    )
    return (combined,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 50 · What this fixes, and cannot be tuned away

    At equal length, a three-phase segment lives $3^{-1/k}$ as long as a
    single-phase one. That ratio depends on the shape alone — no choice of
    scale reaches it, because scale cancels — so **single-phase laterals
    outlive three-phase segments in this model by construction**, not by an
    assumption about lateral cable.

    It reads as a bug the first time the population is inspected, and it is
    worth knowing it is not one, and knowing how large it is: pushing laterals
    below three-phase segments would need a driver the model does not carry.
    """
    )
    return


@app.cell
def _(np, plt, settings):
    _shapes = np.linspace(1.5, 9.0, 100)
    _ratio = 3.0 ** (-1.0 / _shapes)

    _configured = sorted(
        {t.weibull.shape for t in settings.population.technologies}
        | {c.weibull_shape for c in settings.population.classes if c.weibull_shape}
    )

    _figure, _axis = plt.subplots(figsize=(6.5, 3.4))
    _axis.plot(_shapes, _ratio)
    for _s in _configured:
        _axis.plot(_s, 3.0 ** (-1.0 / _s), "o")
        _axis.annotate(
            f"k={_s}: {3.0 ** (-1.0 / _s):.2f}×",
            (_s, 3.0 ** (-1.0 / _s)), textcoords="offset points",
            xytext=(6, -10), fontsize=8,
        )
    _axis.set(
        xlabel="shape k", ylabel="three-phase life ÷ single-phase life",
        title="the min-of-3 penalty, at equal length",
    )
    _axis.axhline(1.0, color="grey", lw=0.8, ls=":")
    _figure.tight_layout()
    _figure
    return


@app.cell
def _(np, settings):
    # Even a very sharp wear-out leaves three-phase segments shorter-lived.
    assert 3.0 ** (-1.0 / 9.0) < 1.0
    _configured = [t.weibull.shape for t in settings.population.technologies]
    _worst = max(3.0 ** (-1.0 / _s) for _s in _configured)
    assert _worst < 1.0
    f"at the configured shapes the penalty never exceeds {_worst:.3f}× — always below 1"
    return


if __name__ == "__main__":
    app.run()
