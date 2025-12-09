#!/bin/bash

INSTANCE_ID=$(vastai show instances | grep running | head -n 1 | awk '{print $1}')
read -p "Is $INSTANCE_ID Instance ID? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Instance ID confirmation failed. Exiting."
  exit 1
fi

if [ ! -f /root/.kaggle/kaggle.json ] || [ ! -f /root/.wandb/wandb.json ]; then
  echo "Error: Required API keys not found"
  exit 1
fi

echo "Installing Python dependencies..."
pip install -r requirements_vastai.txt

if [ ! -d "data" ]; then
  echo "data directory is empty. Downloading data..."
  mkdir -p data
  pushd data
  kaggle datasets download lunran/icrgw2023-data
  unzip icrgw2023-data.zip
  popd
else
  echo "data directory is not empty. Skipping download."
fi

echo "Starting training..."
python -m base_train 2>&1 | tee train.log || true

echo "Starting model upload..."
./tools/upload_model.sh || true

echo "Stopping instance..."
vastai stop instance $INSTANCE_ID
