#!/bin/bash
# kagglehub（0.3.12）のdataset_uploadは、上書きを許可しない

DIR_NAME="upload_code"
mkdir -p ${DIR_NAME}

zip -r ${DIR_NAME}/icrgw2023-code.zip run src_inference1

echo '{
  "title": "icrgw2023-code",
  "id": "lunran/icrgw2023-code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}' > ${DIR_NAME}/dataset-metadata.json

#kaggle datasets create -p ./${DIR_NAME}   # 初回
kaggle datasets version -p ./${DIR_NAME} -m ""   # 2回目以降
