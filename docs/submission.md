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

## Cycle 5 result (contact override + particle filter, 2026-08-03)

Ported from the public physics kernel
[`evgendvorkin/rogii-physics-lb-7-872-v48`](https://www.kaggle.com/code/evgendvorkin/rogii-physics-lb-7-872-v48).
Two submissions this cycle:

1. **Guarded contact override** (kernel v6, ref `55212063`). Reconstructs the
   three visible test wells to ~0.01 ft prefix RMSE, but scored **44.456** —
   identical to the cycle-4 offset trend. Diagnosis: Kaggle rescoring uses a
   hidden test set whose wells have no same-id train copy, so the override
   never fires there and the fallback is what the leaderboard measures.
2. **Particle-filter fallback** (kernel v7, ref `55214209`). Replaces the
   offset trend on hidden wells with a likelihood-weighted particle-filter
   ensemble (see `docs/champion-selection.md`). Toe-holdout confirm RMSE
   **11.225** vs the offset trend's **43.292** over 156 wells. Public score
   **8.752** (down from `44.456`); official team rank `5,465 → 3,231`. This is
   the external confirmation that the fallback path is what the hidden-test
   leaderboard scores, and that the particle filter is a real improvement over
   the offset trend. The remaining gap to the reference kernel's `7.872` is its
   ML stacking + blend + gold-calibration layers, which were not ported.

Both kernels keep the editor-run (visible-well) `submission.csv` byte-identical
because the visible wells resolve through the contact override in either case;
the particle filter only changes the hidden-test predictions the leaderboard
scores. Kernel v7 status `COMPLETE`; the executed Kaggle output byte-matches the
committed artifact on the visible wells.

## Cycle 1 (new registry, SOT-2370) — converge-mode finalization

Deadline `2026-08-05T23:59:00Z` (~2 days out) put this cycle in **converge mode**
(`design/README.md` §51): no new improvement axes, no risky large changes;
finalize verified candidates and select the final submission with recorded
reasoning.

**Final submission champion = the cycle-5 likelihood-weighted particle-filter
fallback** (`champion.json`, kernel v7, submission ref `55214209`):

- Verified: toe-holdout confirm RMSE **11.225** vs the offset trend's **43.292**
  over 156 wells; externally confirmed public LB **8.752** (rank `5,465 → 3,231`).
  It is the best-ever result for this lineage and the hidden-test path the
  leaderboard actually scores.
- Risk diversification note: the guarded contact-override layer (ref `55212063`,
  LB `44.456`) is retained above the PF fallback for the visible wells, so the
  editor-run artifact stays byte-identical; only the hidden-test predictions —
  which the leaderboard scores — resolve through the particle filter.

**No new artifact this cycle.** `champion.json` is unchanged since cycle 5 and
the submitted kernel v7 already carries this champion, so re-submitting would be
byte-identical → recorded as **non-promotion / no new artifact** (per SOT-2370
step 3), not a submission success. The cron and this cycle therefore do not
re-submit.

**Deferred (rejected in converge mode):** closing the remaining `8.752 → 7.872`
gap to the reference kernel requires porting its ML stacking + blend +
gold-calibration layers — a large-model / retraining change forbidden this close
to the deadline. Logged in `docs/ai/experiment_ledger.jsonl` as the next-rung
escalation candidate for a future (post-deadline / non-converge) cycle.

## Cycle 2 (new registry, SOT-2375) — converge-mode re-confirmation (no new artifact)

Still **converge mode** (deadline `2026-08-05T23:59:00Z`, ~2 days out;
`design/README.md` §51): no new improvement axes, no risky large changes.

**Decomposition judgment: not needed.** Cycle 1 (SOT-2370) already finalized the
champion, and the only catalogued next-rung escalation — porting the reference
kernel's ML stacking + blend + gold-calibration to close the `8.752 → 7.872`
gap — is already recorded as **rejected in converge mode** in
`docs/ai/experiment_ledger.jsonl`. Retrying a rejected axis without new evidence
is forbidden, and converge mode forbids opening a new large-model / retraining
axis this close to the deadline. There is therefore no promotable axis to
decompose into child issues this cycle.

**Final submission champion = unchanged**: the cycle-5 likelihood-weighted
particle-filter fallback (`champion.json`, kernel v7, submission ref `55214209`,
public LB **8.752**, rank `5,465 → 3,231`), with the guarded contact-override
layer retained above it for the visible wells (risk diversification).

**No new artifact this cycle → non-promotion.** `champion.json` is unchanged
since cycle 5. The submission gate
(`scripts/ai/kaggle_targets_submit.sh --competition rogii --repo rogii-claude`,
dry run) reports `skip (repeat requires a new artifact fingerprint; selected
artifact was already submitted)` — the current `submission.csv` fingerprint
byte-matches the already-submitted ref `55214209`. Per SOT-2375 step 3 this is
recorded as **non-promotion / no new artifact**, not a submission success; the
cron and child issues do not submit either.

## Cycle 3 (new registry, SOT-2387) — converge-mode re-confirmation (no new artifact)

Still **converge mode** (deadline `2026-08-05T23:59:00Z`, ~2 days out;
`design/README.md` §51): finalize the verified candidate, do not open new
improvement axes, and do not start risky large-model / retraining changes.

**Decomposition judgment: not needed.** This is the third consecutive
converge-mode re-confirmation. The champion was finalized in cycle 1 (SOT-2370)
and re-confirmed in cycle 2 (SOT-2375); nothing about the state has changed:

- `champion.json` is unchanged since cycle 5 (guarded contact-override →
  likelihood-weighted particle-filter fallback → offset-trend).
- The only catalogued next-rung escalation — porting the reference kernel's
  (`evgendvorkin/rogii-physics-lb-7-872-v48`) ML stacking + blend +
  gold-calibration layers to close the `8.752 → 7.872` gap — is a large-model /
  retraining change and is already recorded as **rejected in converge mode** in
  `docs/ai/experiment_ledger.jsonl`. Retrying a rejected axis without new
  evidence is forbidden, and converge mode forbids opening it this close to the
  deadline.
- **Web investigation (this cycle):** searched the competition's public
  solutions for a low-risk, portable win over the current champion. The best
  public approach found (a DWT-based kernel, ~`9.251`) is *worse* than the
  champion's `8.752`; the only stronger published direction is the same ML
  stacking family, which requires large retraining. No new promotable axis
  exists.

**Final submission champion = unchanged**: the cycle-5 likelihood-weighted
particle-filter fallback (`champion.json`, kernel v7, submission ref `55214209`,
public LB **8.752**, rank `5,465 → 3,231`), with the guarded contact-override
layer retained above it for the visible wells (risk diversification).

**No new artifact this cycle → non-promotion.** The submission gate
(`scripts/ai/kaggle_targets_submit.sh --competition rogii --repo rogii-claude`,
dry run) reports `skip (repeat requires a new artifact fingerprint; selected
artifact was already submitted today)` — the current `submission.csv`
fingerprint byte-matches the already-submitted ref `55214209`. Per SOT-2387
step 3 this is recorded as **non-promotion / no new artifact**, not a submission
success; the cron and child issues do not submit either.

## Cycle 3 (SOT-2387) stage 3 — gold-calibration overlay promoted (SOT-2395)

Final stage of the three-stage port of the reference kernel
`evgendvorkin/rogii-physics-lb-7-872-v48` (LB 7.872): stage 1 (SOT-2393) shipped
the offline ML base predictor, stage 2 (SOT-2394) blended it with the champion
particle filter at the hidden-test fallback, and this stage adds the reference
kernel's final **gold-calibration** layer (`ROGII_GOLD_*` = per-well
visible-prefix self-verified anchor, `src/calibrate.py`).

**Result — promoted.** Leak-free fold-0 toe-holdout confirm (156 wells /
746,360 rows, `weight = 0.75`):

| Predictor | Confirm RMSE | vs blend | vs PF |
| --- | ---: | ---: | ---: |
| gold overlay (SOT-2395) | **11.115** | −0.058 | −0.110 |
| stage-2 blend (SOT-2394) | 11.173 | — | −0.052 |
| PF champion (LB 8.752) | 11.225 | +0.052 | — |

The gold overlay beats the stage-2 blend and both standalone predictors, firing
conservatively on only **36 of 156** wells. `champion.json` fallback layer is
updated to `gold_calibrated_physics_ml_blend`; `submission.csv` is regenerated
and stays **byte-identical** (`sha256 46d09239…`) because the three visible test
wells are fully covered by the guarded contact override — only the hidden-test
trajectory changes. Exec byte-identity between `src/` and the kernel mirror is
enforced by `tests/test_calibrate.py`; `pytest` is green (46 passed).

**Kaggle: not submitted here.** Per SOT-2395 the child does not submit; the
parent SOT-2387 owns final-submission selection via
`scripts/ai/kaggle_targets_submit.sh`. This closes the three-stage kernel port
toward the `8.752 → 7.872` gap.

## Cycle 3 (SOT-2387) parent aggregation — kernel v8 submitted

The parent resume run (SOT-2387) aggregated the three completed children and
submitted the new champion. All prerequisite children reached **Done**:

| Stage | Issue | Layer | Leak-free fold-0 confirm RMSE | Decision |
| --- | --- | --- | ---: | --- |
| 1 | SOT-2393 | offline ML base (TVT) predictor | 17.645 | foundation (inconclusive) |
| 2 | SOT-2394 | physics×ML gated blend (`w=0.75`) | 11.173 | promoted (beats PF 11.225) |
| 3 | SOT-2395 | gold-calibration overlay | **11.115** | promoted (beats blend + both singles) |

**Why a new kernel push was required.** This is a Kaggle **code competition**:
the hidden test set is scored by re-running the submitted kernel on Kaggle.
Children are forbidden from submitting, so the on-Kaggle kernel was still
**v7** (cycle-5 PF fallback = public LB 8.752) even though the children had
updated `kaggle/rogii-claude-baseline.py` on disk with the blend + gold layers.
The local `submission.csv` (`sha256 46d09239…`) is byte-identical across cycles
because it only contains the 3 **visible** test wells, all covered by the guarded
contact override — only the **hidden-test fallback path** changed. So the visible
CSV fingerprint is not a sufficient artifact identity for this code competition;
the real artifact is the kernel version.

**Submission (parent-only, via control-plane).**

- Quality gate: `pytest` **46 passed** (incl. `src/`↔kernel exec byte-match for
  ML predictor, blend, and gold-calibration; `__file__`-independent loader).
- Pushed kernel **version 8** (`kaggle kernels push`) → run status COMPLETE
  (interactive run reproduces the 3 visible wells via contact override; the
  gold-calibrated physics-ML blend runs on the hidden toe at submit time).
- Bumped registry `submit.version` 7 → 8 and submitted with
  `bash scripts/ai/kaggle_targets_submit.sh --competition rogii --repo rogii-claude --execute`
  (mandatory path; the Kaggle CLI was **not** called directly to bypass the gate).
- Result: submission **ref 55233567**, `SubmissionStatus.PENDING` (hidden-test
  scoring on this competition takes hours — the concurrent `physics-full` push
  from 01:50 was still PENDING at submit time). The public score is captured by
  the later cron / `collectImproveContext` score-sync once COMPLETE.

The previous champion (kernel v7, ref 55214209, LB **8.752**) remains on the
leaderboard; Kaggle keeps every submission, so this push cannot regress the
selected final score. Deadline `2026-08-05T23:59Z`; 3 submissions remaining today.

## Cycle 6 logic-only reference-NB port — SP45 selector rejected (SOT-2421)

The portable, logic-only remainder of the reference notebook was evaluated as
an SP45-inspired per-well selector over particle-filter likelihood temperature
and conservative heel hold. The selector thresholds and variant map were copied
from the reference profile rather than tuned on the confirmation fold.

**Result — not promoted.** Leak-free fold-0 toe-holdout confirm (156 wells /
746,360 rows) produced candidate gold RMSE **11.541**, worse than the current
champion threshold **11.115** by **0.426**. Its component metrics were blend
11.571 and PF 11.131. Per the non-promotion gate, the candidate implementation
and tests were reverted; `champion.json`, the kernel, and `submission.csv`
remain unchanged.

The reference notebook's GPU-trained LGBM/CatBoost weights and model-package
correction are not portable logic and remain outside this child issue. No
Kaggle submission was made; submission ownership remains with parent SOT-2406.

## Cycle 6 model-package pretrained-ensemble integration — rejected (SOT-2422)

Child SOT-2422 evaluated integrating the reference notebook's **model package**
(`pilkwang/rogii-model-package`, submission profile `vp_balanced_modelpkg_005`)
as offline-attached, inference-only pretrained weights to reach bronze
(`score ≤ 6.410`, ~top 10% of 6,101 teams). The package and its use inside the
reference kernel were pulled and analysed directly (`kaggle datasets`
manifest/blend-config/feature-builders + the reference kernel's
`model_package_correction` cell).

**What the model package actually is.** A `drift_ncc` 5-model stack predicting a
`delta` to `last_known_TVT`, blended (groupkfold OOF weights) as:

| Branch | Type | File | Blend weight |
| --- | --- | --- | ---: |
| catboost_alltrain | `catboost_cbm` | `.cbm` | 0.450 |
| sequence_tcn tcn_residual | **`torch_tcn`** (GPU-trained NN) | `.pt` | 0.417 |
| hgb_alltrain | `sklearn_pickle` (HistGBR) | `.joblib` | 0.118 |
| lgb_alltrain | `lightgbm_booster` | `.txt` | 0.015 |
| xgb_alltrain | `xgboost_json` | `.json` | ~1.6e-13 |

Features are **480 columns** (`drift_ncc_v1`) built by a 146 KB pandas feature
engine (`rogii_feature_core.py`: formation KNN imputers, beam features,
row-ANCC, typewell matching). The package's own group-k-fold OOF RMSE is
**10.670** (3.78 M rows). It is applied to the base submission as a **gated
correction**: `MODEL_PACKAGE_GATED_MAX_WEIGHT = 0.008`,
`gate = gmax / (1 + (|pkg − base| / 6)²)`, auto-disabled when the p95 diff > 25.
On the reference base (public LB **6.477**), the human measured
`vp_balanced_modelpkg_005 ≈ 6.410` — i.e. the model package contributes only
**~0.067 LB (≈1 % relative)**.

**Result — not promoted.** Two independent reasons:

1. **Base gap dominates.** Our on-Kaggle base is kernel v8
   `gold_calibrated_physics_ml_blend` at public LB **8.739**. The 2.329-LB gap to
   bronze is the reference *physics/logic* pipeline, whose port (SP45 selector)
   was already rejected on toe-holdout in SOT-2421 (cand 11.541 > champ 11.115).
   A 0.008-gated ~0.067-LB nudge cannot close a 2.3-LB gap (best case ~8.67).
   The package's `delta` models also assume the reference base's 480-feature
   frame / `last_known_TVT`, which our numpy-only pipeline does not produce — so
   grafting only the package puts it out-of-distribution.
2. **Not portable in practice.** Our submission kernel is stdlib + numpy only.
   Integration would require pandas / scipy / sklearn / lightgbm / catboost /
   xgboost **plus torch** (the TCN carries 42 % of the blend weight and the
   reference kernel runs `enable_gpu: true`), the 146 KB / 480-feature engine,
   and a ~200 MB attached dataset. Dropping the TCN to stay CPU-light removes
   42 % of the ensemble. A high-risk wholesale rewrite the day before the
   deadline — risking the working champion v8 — is not justified by a ~0.067-LB
   expected gain.

`champion.json`, the kernel, and `submission.csv` remain unchanged; `pytest`
stays **46 passed** (the change is docs + ledger only). No Kaggle submission was
made (submission ownership stays with parent SOT-2406).

**Escalation-ladder implication.** The external-knowledge rung (reference-NB
physics base + model package) is now effectively exhausted for reaching bronze
in-repo: bronze requires porting the reference *physics base* itself, which
SOT-2421 could not promote. Whether to flip the target to `mode: "maintain"`
is deferred to parent SOT-2406's deadline-converge run.

## Cycle 9 GR conditioning (detrend + SavGol) & WLS formation calibration — rejected (SOT-2443)

Child SOT-2443 evaluated porting the portable signal-processing techniques from
the public top notebook `romantamrazov/rogii-super-solution-lb-top-3` into the
hidden-test particle-filter fallback (`src/physics.py`), numpy-only and
`__file__`-independent:

1. **GR detrend residual** — remove a robust (IRLS) MD/TVT-linear background trend
   from the GR, applied consistently to the horizontal GR (over MD) and the
   typewell GR (over TVT), so the likelihood matches the local wiggle signature
   rather than each log's slowly-varying baseline.
2. **Savitzky-Golay (window 17, poly 3)** smoothing as a precomputed constant FIR
   filter (coefficients derived from the least-squares polynomial design at
   import; scipy-free).
3. **WLS (exp-decay 0.02) tail-upweighted** estimate of the stratigraphic offset
   `U = TVT + Z` and its slope, seeding the particle cloud and drift — the
   reference kernel's tail-weighted formation `b_well` calibration (our
   hidden-test path has no formation column, so it maps to the PF anchor/drift).

The `src` and kernel copies were kept numerically identical (`predict_pf_well`
`np.array_equal` parity verified on a screen well).

**Result — not promoted (large regression).** The gate is the leak-free fold-0
toe-holdout `gold` RMSE vs the current champion (**11.074**, the SOT-2442
beam+NCC pool). The candidate fails the mandatory screen pre-gate decisively:

| Stage (fold-0, leak-free) | Champion | SOT-2443 candidate |
| --- | ---: | ---: |
| screen `gold` RMSE (5 wells) | 5.778 | **35.867** (blend 37.096 / pf 45.642) |
| confirm `gold` RMSE (156 wells / 746,360 rows) | 11.074 | **23.452** (blend 24.130 / pf 28.074) |

A per-component isolation of the **PF-only** pooled screen RMSE (5 wells) shows
*every* piece regresses the already-tuned champion filter — nothing is salvageable:

| PF configuration | pf screen RMSE |
| --- | ---: |
| champion (raw GR, median drift) | **2.980** |
| WLS only (raw GR) | 5.327 |
| SavGol only (no detrend) + median drift | 38.175 |
| SavGol + detrend + median drift | 41.073 |
| SavGol only (no detrend) + WLS | 36.693 |
| full (SavGol + detrend + WLS) | 45.642 |

**Root cause.** The champion PF localizes TVT by matching the *raw* GR wiggle
against the typewell GR-vs-TVT signature. Savitzky-Golay smoothing removes exactly
the high-frequency marker features the likelihood relies on (2.98 → 38.2, a ~13×
regression on its own); detrending the two logs over different depth axes
(MD vs TVT) with different sampling further decorrelates them; and the exp-decay
tail-weighted anchor/drift is worse than anchoring at the last known row with a
median heel slope (2.98 → 5.33). This matches the a-priori: all four public
notebooks (LB 9.251–10.811) already score *worse* than our champion (LB 8.739),
so their conditioning does not transfer to our tuned pipeline.

`src/physics.py`, `kaggle/rogii-claude-baseline.py`, `champion.json`, the kernel,
and `submission.csv` remain **unchanged** (implementation reverted); `pytest`
stays green (the change is docs + ledger only). No Kaggle submission was made
(submission ownership stays with parent SOT-2438).
