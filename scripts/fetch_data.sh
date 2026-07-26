#!/usr/bin/env bash
set -euo pipefail

competition="${KAGGLE_COMPETITION:-rogii-wellbore-geology-prediction}"
destination="${1:-data/raw}"

command -v kaggle >/dev/null 2>&1 || {
  echo "Kaggle CLI is required: https://github.com/Kaggle/kaggle-api" >&2
  exit 1
}

mkdir -p "$destination"
kaggle competitions download -c "$competition" -p "$destination"
archive="$destination/$competition.zip"
unzip -q "$archive" 'sample_submission.csv' 'train/*.csv' 'test/*.csv' -d "$destination"
echo "Data extracted to $destination (downloaded archive retained locally)"
