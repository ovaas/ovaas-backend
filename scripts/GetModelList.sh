#!/bin/bash
openvino_version=$1
docker run --rm openvino/ubuntu18_dev:$openvino_version /bin/bash -c "python3 /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py --print_all"
