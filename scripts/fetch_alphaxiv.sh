#!/usr/bin/env bash
# Fetch the two files we use unchanged from alphaXiv's unofficial repro of the paper
# (github.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief). The repo has no license, so
# they are not redistributed here: this downloads them at the pinned commit and checks their sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=1d9ec4311fd4137652034077d7f7e0fe78a24916
URL=https://raw.githubusercontent.com/alphaXiv/measuring-reward-seeking-via-contrastive-belief/$COMMIT/src
DEST=src/rsr/alphaxiv
mkdir -p "$DEST"
fetch() {  # file sha256
  curl -fsSL "$URL/$1" -o "$DEST/$1"
  echo "$2  $DEST/$1" | shasum -a 256 -c -
}
fetch classify.py 2fac24e39d6a1b687b1f27a8d6c29e0567d77143c8076bede64e830031eb8639
fetch tasks.py e95264191b561dd6d8fe25726929039b6003306c16887f3cb2e0e4c24d48aa20
: > "$DEST/__init__.py"
