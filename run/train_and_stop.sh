#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Instance ID is required"
  echo "Usage: $0 <instance_id>"
  exit 1
fi

INSTANCE_ID=$1

uv run python -m base_train 2>&1 | tee train.log || true
vastai stop instance $INSTANCE_ID
