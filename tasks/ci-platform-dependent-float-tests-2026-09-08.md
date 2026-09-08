# The required check goes red on some runners, on code that is fine

## What is happening

Four tests fail intermittently in the `test` GitHub check. They pass locally,
serial and under `-n auto`, and they pass on the same commit on a later run.

| test | what it asserts |
|---|---|
| `test_numpy_and_the_system_library_agree_in_single_precision` | NumPy and the system C library return the same bits for single-precision `expm1`, `log1p` and `pow` |
| `test_the_emergency_premium_is_narrowed_where_the_reference_narrows_it` (both cases) | a searched budget funds exactly 181 candidates, so it straddles two cut points |
| `test_the_premium_comment_quotes_a_multiplier_that_moves_what_it_says` | a multiplier quoted in a Rust comment moves exactly 8,883 premium terms and 7,012 rank keys |

**It is the machine, not the diff.** Commit `53a3e2c` was run three times
through the same workflow: red, red, then green, on runner image
`ubuntu-24.04`, provisioner `20260828.587` every time. `main` at `1257669` went
red once and green three times the same day with all three test files
unchanged. The single-precision agreement test reaches the C library through
`ctypes` and a seeded generator, so nothing in any recent diff can touch it.

**The single-precision agreement test is the root, and the other two follow
it.** Across the last sixty runs of this workflow there are three distinct red
runs. That test fails in all three; the two knife-edge parity tests fail in two
of them, and in the third it failed alone. So the ordering is: the platform's
NumPy-against-C-library agreement breaks, and where the break is large enough
it also moves the searched constants the parity fixtures sit on.

**The red rate is understated by the run list.** Re-running a job overwrites
its conclusion, so a run that went red and was re-run to green reads as a
success afterwards. Three visible reds in sixty runs is a floor, not the rate.

## What is not known, and must be found before anything is changed

**The mechanism.** The obvious explanation is that one side dispatches on CPU
features and the runner fleet is heterogeneous. That is a hypothesis and it did
not survive the two tests available locally:

- `NPY_DISABLE_CPU_FEATURES="AVX2,FMA3"` — NumPy's `expm1`, `log1p` and `pow`
  over the test's own sample return **bit-identical** results with and without.
- `GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2_Usable,-FMA_Usable,-AVX_Usable` — the
  C library's `expm1f`, `log1pf` and `powf` likewise **bit-identical**.

This machine has no AVX-512 (NumPy reports `found: X86_V3`,
`not_found: X86_V4, AVX512_ICL, AVX512_SPR`), so the one path that could still
differ is the one that cannot be exercised here. That is a reason to go and
look at a failing runner, not a reason to assume.

**So step one is diagnosis, and no test changes before it.** Every remedy below
depends on which inputs disagree and by how much, and a fix chosen without that
would be a guess dressed as a repair.

## Step 1 — make the next red run say what happened

Add a diagnostic that runs in CI and records, whenever the agreement test
fails, everything needed to explain it without a second red run:

- `/proc/cpuinfo` model name and the flag set, `ldd --version` for the C
  library, the NumPy version, and NumPy's `simd_extensions` from
  `np.show_runtime()`.
- **The disagreements themselves**: for each of `expm1`, `log1p` and `pow`, how
  many of the 20,000 inputs differ, the largest difference in units in the last
  place, and the first few offending input values with both results.

The last of those is what decides the remedy, so it is the part not to skip.
The question it answers: **do the disagreeing inputs lie in the range the
annual loop actually reaches?**

Where it goes matters. A failing assertion prints nothing extra, so this is
either a `pytest` failure message built from the comparison it already
computes — the cheapest option, since the test holds all three arrays already —
or a separate always-run CI step that prints the fingerprint whatever happens.
Prefer building it into the assertion message: it then reports the same way for
someone running locally on a machine that disagrees.

This step is worth doing on its own even if the remedy turns out to be
straightforward, because the same information is what tells us whether a future
red is this problem or a new one.

## Step 2 — the remedy, chosen from what step 1 finds

Three shapes, and which is right depends entirely on the disagreeing inputs.

**If the disagreements fall outside the range the loop reaches.** The test's
docstring says each quantity is sampled "over the range the annual loop
actually reaches it in" — the hazard exponent over a plausible age-to-scale
ratio, the lifetime draw over the open unit interval. If the offending inputs
sit outside what the shipped population and configuration actually produce,
then the sample is broader than the property, and narrowing it to the real
range makes the test both correct and stable. **This is the outcome to hope
for, and the first thing to check.** The catch is written into the test
already: its second assertion — that the answers are not merely double
arithmetic rounded once — has only a 15-in-20,000 margin on the `pow` arm, so
narrowing the range or shrinking the sample can disarm that arm while the
others go on passing. Any narrowing has to be checked against that margin, and
the check has to be watched failing.

**If the disagreements fall inside that range.** Then single-precision runs are
not portable, and that is a finding about the project rather than a test to
loosen. The whole f32 comparison story rests on both sides reaching the same C
library; where they do not, exact parity at f32 is not available. Note that on
the red runs the f32 parity tests **passed** — 485 of them — so the simulation's
own inputs did not reach a disagreeing value even on a machine where the sample
did. That is luck of the inputs rather than a guarantee, and it is the
distinction this case turns on. The remedy is then a stated platform
precondition: one predicate, checked once, with the f32 exactness tests
skipping and saying why where it does not hold, and the fact recorded in the
project's decisions rather than buried in a skip.

**Either way, the two knife-edge parity fixtures need their own decision.**
They are deliberately at the edge — the emergency-bill sibling straddles
79,962,480 against 79,962,488, eight dollars in eighty million, which is single
precision's epsilon. A fixture placed there cannot be made robust to a
one-unit-in-the-last-place platform difference without giving up what it
pins. Their guards already say "re-search against this population", and the
guard fired correctly; it simply named the wrong cause. Two options:

- **Skip them under the same platform predicate**, which is honest — they test
  single-precision boundary behaviour and are meaningless where the platform
  assumption fails — and keeps the required check green.
- **Re-search each budget to sit mid-way between its two cut points** rather
  than adjacent to one. This preserves the test unchanged and needs no new
  concept, but only helps if the gap between the cut points is wider than the
  platform difference. Step 1's largest-difference figure says whether it is.

Prefer the second where the numbers allow it, because it keeps a test that runs
everywhere over a test that is skipped somewhere.

## What not to do

- **Do not add a tolerance to the agreement test.** It exists precisely because
  this project compares implementations at no tolerance; a version of it that
  tolerates a one-unit difference asserts nothing the parity suite does not
  already assume.
- **Do not delete or mark the tests as expected failures to get the check
  green.** All three guard a real property, and the check going red is them
  working. What is wrong is that a merge is blocked by a property of the
  runner, not that the property is checked.
- **Do not move them out of the required check** without deciding the question
  above first. A platform check that nothing reads is a platform check that has
  been removed.

## Verification

- **Watch each changed test fail.** Whatever the remedy, the property each test
  currently guards must still be pinned: mutate the arithmetic each one
  protects and confirm it still reddens. For the agreement test that means the
  double-then-round path it rules out; for the parity fixtures, the premium
  narrowing order and the multiplier the Rust comment quotes.
- **A skip must be watched skipping and watched not skipping.** If the remedy
  is a platform predicate, force it both ways and confirm the suite reports
  each — a predicate that is always true is a skip that never fires and a test
  that never runs.
- **Re-run the check several times after the fix.** One green run proves
  nothing here; that is how this was misdiagnosed twice on the way in. The
  runner that produces the failure appears in a minority of runs, so the
  evidence for a fix is a run of greens long enough to be worth something,
  and the honest report says how many.

## Open questions for the user

1. **Is single precision meant to be portable?** If the answer from step 1 is
   that the platform genuinely disagrees on inputs the loop reaches, this is
   the decision that follows, and it belongs in the project's own record of
   decisions rather than in a test.
2. **Is a skipped test acceptable in this suite?** There is currently no
   platform-conditional skip anywhere in it, so this would be the first, and it
   is the kind of thing that quietly spreads.
