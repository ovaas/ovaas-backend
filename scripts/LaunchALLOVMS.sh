#!/bin/bash

#specifiy the target directory
dir="$(pwd)/ovms"

#check if the directory exists
if [ ! -d $dir ]; then
    mkdir $dir
fi

# model server ports
ports=("9000" "9001" "9002" "9003" "9004" "9005" "9006" "9007" "9008" "9009")

# model names
models=("human-pose-estimation"
        "handwritten-japanese-recognition"
        "colorization"
        "objectdetection"
        "semantic-segmentation-adas"
        "midasnet"
        "yolo-v3"
        "face-detection"
        "age-gender-recognition"
        "super-resolution")

# All sercers have same "model server version" & "ip address"
MODEL_SERVER_VERSION="latest"
IP_ADDRESS=$1

# All servers are going to be started
for num in {0..9} ; do
    PORT_NUMBER=${ports[$num]}
    MODEL_NAME=${models[$num]}
    MODEL_PATH="az://ovms/$MODEL_NAME"
    AZURE_STORAGE_CONNECTION_STRING="AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;DefaultEndpointsProtocol=http;BlobEndpoint=http://$IP_ADDRESS:10000/devstoreaccount1;QueueEndpoint=http://$IP_ADDRESS:10001/devstoreaccount1;TableEndpoint=http://$IP_ADDRESS:10002/devstoreaccount1;"
    docker run --rm -d -v $dir:/log -p $PORT_NUMBER:9000 -e AZURE_STORAGE_CONNECTION_STRING=$AZURE_STORAGE_CONNECTION_STRING openvino/model_server:$MODEL_SERVER_VERSION --model_path $MODEL_PATH --model_name $MODEL_NAME --port 9000 --log_level DEBUG --log_path "/log/$MODEL_NAME.log" --file_system_poll_wait_seconds 0
done
