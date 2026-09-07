//! Weibull lifetimes: the annual failure probability, and the two draws.
//!
//! Two derived forms of the Weibull are used, for two different jobs. Sampling
//! a lifetime inverts the survivor function; scoring a replacement candidate
//! uses the discrete annual failure probability. A policy ranks on the second
//! because that is what a planner knows — it never sees the sampled failure
//! time.
//!
//! The Rust side of the mirror: `python/cablesim/weibull.py` carries the same
//! three functions under the same names, and the parity tests between them are
//! what validate this side. The effective-scale reduction and the censored fit
//! live only in Python: each runs once, before anything crosses the boundary,
//! so neither is in this loop.
//!
//! # Reading this beside the Python
//!
//! The Python functions take whole NumPy arrays and return whole arrays; these
//! take one `f64` and return one `f64`, and the caller loops. That is the one
//! systematic difference between the two files, and it is not a translation
//! artefact: NumPy is fast because the loop is inside the C it calls, so Python
//! is written to hand it as much work per call as possible. A Rust loop over
//! scalars compiles to the same machine code the array version would, so the
//! scalar form costs nothing here and reads closer to the formula.
//!
//! Three pieces of syntax carry most of the difference:
//!
//! * `x.powf(y)` is Python's `x ** y`. Rust has no exponent operator, and
//!   `^` means bitwise exclusive-or, so writing `x ^ y` compiles for integers
//!   and silently means something else.
//! * `.exp_m1()` and `.ln_1p()` are NumPy's `expm1` and `log1p`, and are used
//!   for the same reason: near zero they keep the precision that `exp(x) - 1`
//!   and `ln(1 + x)` lose. Written as methods on the value rather than as
//!   functions taking it, which is Rust's usual shape for anything numeric.
//! * `pub fn` is a function visible outside this file. Without `pub` it would
//!   be private to the module, which is Rust's default — the opposite of
//!   Python, where a leading underscore is a convention the interpreter does
//!   not enforce.

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
/// hazard accumulated *within the year* — the bracketed difference above, not
/// the cumulative hazard to `age` — passes 37.4299, where `exp(-h)` falls
/// below half an ulp of 1 and the subtraction rounds to 1. No segment the
/// shipped population generates reaches that: the within-year hazard crosses
/// 37.4299 somewhere between ages 105 and 222 across the 12,000 shipped
/// segments, against a maximum age of 89 scored inside a 30-year horizon, and
/// the largest value this returns anywhere in that horizon is 0.9999954. The
/// closed upper bound is what the interval says rather than something the loop
/// exercises.
pub fn conditional_failure_probability(age: f64, shape: f64, scale: f64) -> f64 {
    // The hazard accumulated over this one year: a difference of two cumulative
    // hazards rather than a cumulative hazard itself. Naming it for the
    // accumulation and not for the year is what sent two attempts at the note
    // above to measure the wrong quantity.
    let annual_hazard = ((age + 1.0) / scale).powf(shape) - (age / scale).powf(shape);
    // The last expression in a Rust function is its return value, with no
    // `return` keyword and no semicolon — a semicolon here would discard the
    // value and return the empty tuple instead, which is what Rust uses where
    // Python returns `None`.
    -((-annual_hazard).exp_m1())
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
/// accumulated back in: `T = scale * ((age/scale)^k - ln u)^(1/k)`, and the
/// remaining life is `T - age`.
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
    // `let` binds a name. Bindings are immutable unless written `let mut`,
    // which is the reverse of Python's default and is why most of this crate
    // has no `mut` on it: a name that is never reassigned says so in the
    // declaration rather than in a comment.
    let accumulated = (age / scale).powf(shape);
    let total = scale * (accumulated - (-u).ln_1p()).powf(1.0 / shape);
    total - age
}

// `#[cfg(test)]` compiles this module only under `cargo test`, so the test code
// is absent from the shipped extension rather than merely unreached. Rust puts
// a module's unit tests in the same file as the code, which is why there is no
// `tests/weibull.rs` to match `tests/test_weibull.py`; the Python suite's
// equivalents live under `tests/` because that is where pytest looks.
#[cfg(test)]
mod tests {
    // `super` is the enclosing module — this file. The glob import brings its
    // three functions into scope, since a child module does not inherit its
    // parent's names the way a nested Python scope does.
    use super::*;

    /// The annual probabilities compound into the survivor function.
    ///
    /// Deterministic, so it is asserted exactly rather than statistically: the
    /// product of `1 - p(t)` over a horizon equals `S(horizon)`.
    #[test]
    fn annual_probabilities_compound_into_the_survivor_function() {
        let (shape, scale) = (6.2, 65.0);
        // `(0..30)` is Python's `range(30)`, and `.map(...).product()` is
        // `math.prod(... for year in ...)`. The closure `|year| ...` is a
        // lambda; the vertical bars hold its parameters.
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
    /// parity tests remove randomness, so both are checked here.
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
