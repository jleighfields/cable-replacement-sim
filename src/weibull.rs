//! Weibull lifetimes: the annual failure probability, and the two draws.
//!
//! Two derived forms of the Weibull are used, for two different jobs. Sampling
//! a lifetime inverts the survivor function; scoring a replacement candidate
//! uses the discrete annual failure probability. A policy ranks on the second
//! because that is what a planner knows — it never sees the sampled failure
//! time.
//!
//! The Rust side of the mirror: `python/cablesim/weibull.py` implements these
//! three functions over whole arrays under the same names, and the parity
//! tests between them are what validate this side. The effective-scale
//! reduction and the censored fit live only in Python: each runs once, before
//! anything crosses the boundary, so neither is in this loop.

/// Probability of failing within a year, given survival to `age`.
///
/// This is what a replacement policy ranks on:
/// `p(t) = 1 - S(t+1)/S(t) = 1 - exp(-[((t+1)/scale)^k - (t/scale)^k])`.
///
/// # Arguments
///
/// * `age` - current age, in years.
/// * `shape` - Weibull shape parameter.
/// * `scale` - effective Weibull scale, in years.
///
/// # Returns
///
/// The annual failure probability, on `[0, 1]`. It reaches exactly 1 once the
/// accumulated hazard passes about 745, where `exp` underflows — an age far
/// beyond anything this model simulates, but the bound is closed rather than
/// half-open.
pub fn conditional_failure_probability(age: f64, shape: f64, scale: f64) -> f64 {
    let accumulated = ((age + 1.0) / scale).powf(shape) - (age / scale).powf(shape);
    // `exp_m1` rather than `exp(x) - 1`, matching NumPy's `expm1`: the two
    // differ in the last bits for small hazards, which is every young segment.
    -((-accumulated).exp_m1())
}

/// Inverts the survivor function to sample a lifetime for new cable.
///
/// # Arguments
///
/// * `u` - a uniform on `[0, 1)`.
/// * `shape` - Weibull shape parameter.
/// * `scale` - effective Weibull scale, in years.
///
/// # Returns
///
/// Age at failure, in years.
pub fn draw_lifetime(u: f64, shape: f64, scale: f64) -> f64 {
    scale * (-((-u).ln_1p())).powf(1.0 / shape)
}

/// Samples remaining life for cable that has already survived to `age`.
///
/// The draw is **conditional** on that survival, which adds the hazard already
/// accumulated back in:
/// `T = scale * ((age/scale)^k - ln u)^(1/k)`, and the remaining life is
/// `T - age`.
///
/// Drawing unconditionally here makes a population that starts partway through
/// its life behave as though it were new, which inflates every policy's
/// apparent performance, and it does so silently — the run completes and the
/// curves look plausible.
///
/// # Arguments
///
/// * `u` - a uniform on `[0, 1)`.
/// * `age` - current age, in years.
/// * `shape` - Weibull shape parameter.
/// * `scale` - effective Weibull scale, in years.
///
/// # Returns
///
/// Remaining life from `age`, in years.
pub fn draw_remaining_life(u: f64, age: f64, shape: f64, scale: f64) -> f64 {
    let accumulated = (age / scale).powf(shape);
    let total = scale * (accumulated - (-u).ln_1p()).powf(1.0 / shape);
    total - age
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The annual probabilities compound into the survivor function.
    ///
    /// Deterministic, so it is asserted exactly rather than statistically:
    /// the product of `1 - p(t)` over a horizon equals `S(horizon)`.
    #[test]
    fn annual_probabilities_compound_into_the_survivor_function() {
        let (shape, scale) = (6.2, 65.0);
        let survival: f64 = (0..30)
            .map(|year| 1.0 - conditional_failure_probability(f64::from(year), shape, scale))
            .product();

        let closed_form = (-(30.0f64 / scale).powf(shape)).exp();

        assert!((survival - closed_form).abs() < 1e-12 * closed_form);
    }

    /// A draw conditional on survival to age 0 is an unconditional draw.
    #[test]
    fn remaining_life_at_age_zero_is_a_whole_lifetime() {
        for &u in &[0.01, 0.5, 0.99] {
            let conditional = draw_remaining_life(u, 0.0, 6.2, 65.0);
            let unconditional = draw_lifetime(u, 6.2, 65.0);
            assert!((conditional - unconditional).abs() < 1e-12 * unconditional);
        }
    }

    /// A scale far past the horizon puts the first failure out of reach, and
    /// one near zero puts it at the present. Both are how the deterministic
    /// parity test removes randomness, so both are checked here.
    #[test]
    fn the_scale_bounds_the_deterministic_parity_test_relies_on_hold() {
        assert!(draw_remaining_life(0.5, 40.0, 6.2, 1e6) > 100_000.0);
        // Not asserted as exactly zero: the accumulated hazard at this scale
        // is around 1e47, so the drawn total lifetime is the entry age back
        // again to within a last-bit rounding of `powf`. What the test needs
        // is that the segment fails inside its first year.
        assert!(draw_remaining_life(0.5, 40.0, 6.2, 1e-6) < 1e-6);
    }
}
