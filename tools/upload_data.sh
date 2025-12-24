#!/bin/bash

DIR_NAME="upload_data"
mkdir -p ${DIR_NAME}

pushd "data"  
zip -r ../${DIR_NAME}/archive.zip test train.hdf5 validation.hdf5 resnet18-imagenet.pth
popd

echo '{
  "title": "icrgw2023-data",
  "id": "lunran/icrgw2023-data",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}' > ./${DIR_NAME}/dataset-metadata.json

#kaggle datasets create -p ./${DIR_NAME}   # 初回
kaggle datasets version -p ./${DIR_NAME} -m ""   # 2回目以降
