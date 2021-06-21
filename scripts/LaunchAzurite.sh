#!/bin/bash
#チェック対象のディレクトリを指定
dir="$(pwd)/azurite"

#ディレクトリが存在するかチェックする
if [ ! -d $dir ]; then
    mkdir $dir
fi

docker run -d --rm -p 10000:10000 -p 10001:10001 -v $dir:/data mcr.microsoft.com/azure-storage/azurite
