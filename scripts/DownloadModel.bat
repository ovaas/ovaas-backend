rem �`�F�b�N�Ώۂ̃f�B���N�g�����w��
SET dir=%CD%\models

rem �f�B���N�g�������݂��邩�`�F�b�N����
If not exist %dir% mkdir %dir%

SET model_name=%1
SET openvino_version=%2
docker run --rm -v %dir%:/share openvino/ubuntu18_dev:%openvino_version% /bin/bash -c "cd /share && /opt/intel/openvino_2021/deployment_tools/tools/model_downloader/downloader.py --name %model_name%"
