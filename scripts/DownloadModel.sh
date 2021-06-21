#!/bin/bash
#チェック対象のディレクトリを指定
dir="$(pwd)/models"

#ディレクトリが存在するかチェックする
if [ ! -d $dir ]; then
    mkdir $dir
fi

model_name=$1
docker run --rm -v $dir:/share openvino/ubuntu18_dev:latest /bin/bash -c "cd /share && /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py --name $model_name"
