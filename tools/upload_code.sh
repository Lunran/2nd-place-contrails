#!/bin/bash
# kagglehub（0.3.12）のdataset_uploadは、上書きを許可しない

tar -cvzf icrgw2023-code.tar.gz run src_inference1
mkdir tmp
mv icrgw2023-code.tar.gz tmp/
echo '{
  "title": "icrgw2023-code",
  "id": "lunran/icrgw2023-code",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}' > tmp/dataset-metadata.json
pushd tmp
#kaggle datasets create -p .   # 初回
kaggle datasets version -p . -m ""   # 2回目以降
popd

rm -rf tmp
