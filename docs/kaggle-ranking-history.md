# Kaggle ranking history

- Competition: [ROGII Wellbore Geology Prediction](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction)
- Kaggle team: `sota1111`
- Lineage: Claude

| Observed at (UTC) | Official team rank | Teams | This lineage's observed public score |
| --- | ---: | ---: | ---: |
| 2026-07-26 13:38 | 5,725 | 5,730 | 11551.955 |
| 2026-07-28 15:57 | 5,469 | 5,844 | 11551.955 |
| 2026-07-29 07:52 | 5,465 | 5,885 | 44.456 |
| 2026-08-03 14:30 | 3,231 | 6,084 | 8.752 |

The official rank is shared by the GPT and Claude repositories because both
submit under the same Kaggle team. The latest score is attributed to this
lineage by submission ref `55214209` (kernel v7) and description
`cycle-5 particle-filter fallback champion`. The `44.456 → 8.752` jump
(rank `5,465 → 3,231`) came from the ported particle-filter fallback, which the
hidden-test leaderboard actually scores (the guarded contact override, ref
`55212063`, also scored `44.456` because it only fires on wells with a same-id
train copy — absent from the rescored hidden set). See `docs/submission.md`.

Source: Kaggle CLI `competitions list` and `competitions submissions`. Earlier official-rank snapshots were not retained, so they are not reconstructed or estimated.
