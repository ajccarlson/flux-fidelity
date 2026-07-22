#!/bin/sh
set -eu

export PYTHONHASHSEED=0
export TZ=UTC
export LC_ALL=C.UTF-8

python -m pip install --disable-pip-version-check --ignore-installed --require-hashes \
  -r tools/model-reproduction/requirements-rife-fp16.txt
python tools/model-reproduction/reproduce-rife-fp16.py --check
