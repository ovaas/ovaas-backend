#!/bin/bash
#チェック対象のディレクトリを指定
dir="$(pwd)/ovms"

#ディレクトリが存在するかチェックする
if [ ! -d $dir ]; then
    mkdir $dir
fi

PORT_NUMBER=$3
MODEL_SERVER_VERSION="latest"
MODEL_NAME=$1
MODEL_PATH="az://ovms/$MODEL_NAME"
IP_ADDRESS=$2
AZURE_STORAGE_CONNECTION_STRING="AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;DefaultEndpointsProtocol=http;BlobEndpoint=http://$IP_ADDRESS:10000/devstoreaccount1;QueueEndpoint=http://$IP_ADDRESS:10001/devstoreaccount1;TableEndpoint=http://$IP_ADDRESS:10002/devstoreaccount1;"
docker run --rm -d -v $dir:/log -p $PORT_NUMBER:9000 -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING openvino/model_server:$MODEL_SERVER_VERSION --model_path $MODEL_PATH --model_name $MODEL_NAME --port 9000 --log_level DEBUG --log_path "/log/$MODEL_NAME.log" --file_system_poll_wait_seconds 0
