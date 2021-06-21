#!/bin/bash
docker run --rm openvino/ubuntu18_dev:latest /bin/bash -c "python3 /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py --print_all"
