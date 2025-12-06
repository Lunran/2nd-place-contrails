#!/bin/bash

DIR_NAME="upload_model"
mkdir -p ${DIR_NAME}

pushd "experiments"
zip -r ../${DIR_NAME}/archive.zip CoaT_ULSTM.pth
popd

echo '{
  "title": "icrgw2023-model",
  "id": "lunran/icrgw2023-model",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}' > ./${DIR_NAME}/dataset-metadata.json

#kaggle datasets create -p ./${DIR_NAME}   # 初回
kaggle datasets version -p ./${DIR_NAME} -m ""   # 2回目以降
