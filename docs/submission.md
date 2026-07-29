# Champion submission

SOT-1974 wires the retained `TVT_input`/zero-fallback champion to both a local
CSV generator and the notebook-only Kaggle submission route.

## Verification

- Generated rows: 14,151 across all 3 test wells.
- Format: `id,tvt`, with the same row count, ID set, and row order as
  `sample_submission.csv`.
- Exec gate: passed from an unrelated temporary directory by executing
  `src/predict.py` without defining `__file__`.
- Additional packages: none; both entry points use the Python standard library.

## Kaggle result

- Kernel: `sota1111/rogii-claude-baseline`, version 1, status `COMPLETE`.
- Submission reference: `54995547`, status `COMPLETE`.
- Public score: **11551.955**.
- Comparison: equal to the recorded baseline score of approximately
  `11551.955`, as expected for the retained zero-fallback champion.

Direct file upload was rejected by the Kaggle API because this competition
accepts submissions from Notebooks only. The registry therefore retains
`submit.file` for the generated artifact and also specifies the kernel,
version, and output fields used by the supported submission route.

## Cycle 2 result (SOT-2035)

SOT-2034's constrained TVT correction failed the mandatory screen gate, so
cycle 2 made no promotion. The registered `pass_through_baseline` in
`champion.json` therefore remained the submission champion. Its local generator
and Kaggle entry point produced byte-identical CSV files with 14,151 finite
predictions, exact `id,tvt` columns, and the sample submission's ID set and row
order.

- Kernel: `sota1111/rogii-claude-baseline`, version 2, status `COMPLETE`.
- Submission reference: `55025165`, status `COMPLETE`.
- Public score: **11551.955**.
- Public leaderboard rank at verification time: **5766**.
- Submitted at: `2026-07-27 09:50:56 UTC`.

The unchanged score matches version 1 and is the expected external confirmation
for the retained zero-fallback champion. The cycle-2 result is non-promotion:
no constrained-correction candidate code is present in the inference path.

## Cycle 3 result (SOT-2092 / SOT-2093)

Root cause of the flat rank across cycles 1–2: the retained champion emitted the
sample's zero fallback for every withheld toe row, so all submissions tied at the
all-zero public score of `11551.955`. Cycle 3 promoted the **offset-trend toe
extrapolation** champion (`champion.json`, SOT-2092): per well, fit
`offset = TVT_input + Z` as a linear trend in MD over the heel rows where
`TVT_input` is known, then extrapolate the withheld toe as `tvt = offset(MD) - Z`.

SOT-2093 wired it end-to-end for real scoring:

- `kaggle/rogii-claude-baseline.py` reproduces the champion using only the Python
  standard library (no internet), preferring the competition `test` split when a
  well id appears under both `train` and `test`. Its `/kaggle/working/submission.csv`
  is byte-identical to the local `src/predict.py` output (14,151 finite non-zero
  rows), verified both locally and from the executed Kaggle kernel run.
- Exec gate: `tests/test_kernel.py` runs the kernel from an unrelated cwd with no
  `__file__` defined and asserts byte-for-byte parity with the local generator.

- Kernel: `sota1111/rogii-claude-baseline`, version 3, status `COMPLETE`.
- Submission reference: `55053814`, status `COMPLETE`.
- Public score: **62.332** (down from `11551.955`).
- Public leaderboard rank at verification time: **5447 / 5821** (up from `5766`).
- Submitted at: `2026-07-28 11:39:42 UTC`.

This is a promotion confirmed by external scoring: the offset-trend champion
improves both the public RMSE (`11551.955 → 62.332`) and the rank
(`5766 → 5447`) over the zero-fallback baseline it replaced.

## Cycle 4 result (SOT-2156 / SOT-2158)

Cycle 4 promoted the exponential recency-weighted offset-trend champion with
`recency_decay=8`. Its confirm holdout metrics were RMSE `43.292298` and MAE
`29.559006`. The Theil–Sen alternative evaluated in SOT-2157 did not pass the
screen gate, so it was not promoted.

SOT-2158 regenerated all 14,151 predictions and pushed the updated standard
library-only kernel. Python 3.11 and Kaggle Python 3.12 differed at the final
binary floating-point rounding bit when serializing ten decimals, so both
generators now serialize seven decimals. This changes predictions by less than
`5e-8` while making the executed Kaggle output byte-identical to the local
artifact:

- SHA-256: `d0ec1fc356507cd97bcb7d6e300f2cfc00266ab6c8b6ba0dce571fdc72e99ed6`.
- Kernel: `sota1111/rogii-claude-baseline`, version 5, status `COMPLETE`.
- Submission reference: `55075600`, status `COMPLETE`.
- Public score: **44.456** (improved from cycle 3's `62.332`).
- Public leaderboard rank at verification time: **5465 / 5885**.
- Submitted at: `2026-07-29 07:51:32 UTC`.

The real Kaggle result confirms the recency-weighted cycle-4 promotion.
