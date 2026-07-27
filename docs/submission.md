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
