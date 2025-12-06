#!/bin/bash

DIR_NAME="submit_model"
pushd $DIR_NAME
zip -r ../model/${DIR_NAME}.zip *
popd

echo '{
  "title": "icrgw2023-model",
  "id": "lunran/icrgw2023-model",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}' > ./model/dataset-metadata.json

#kaggle datasets create -p ./model   # 初回
kaggle datasets version -p ./model -m ""   # 2回目以降
