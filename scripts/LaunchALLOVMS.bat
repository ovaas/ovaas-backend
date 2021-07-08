@echo off

rem チェック対象のディレクトリを指定
SET dir=%CD%\ovms

rem ディレクトリが存在するかチェックする
If not exist %dir% mkdir %dir%

@REM model server ports
SET ports[0]=9000
SET ports[1]=9001
SET ports[2]=9002
SET ports[3]=9003
SET ports[4]=9004
SET ports[5]=9005
SET ports[6]=9006
SET ports[7]=9007
SET ports[8]=9008

@REM model names
SET models[0]=human-pose-estimation
SET models[1]=handwritten-japanese-recognition
SET models[2]=colorization
SET models[3]=objectdetection
SET models[4]=semantic-segmentation-adas
SET models[5]=midasnet
SET models[6]=yolo-v3
SET models[7]=face-detection
SET models[8]=age-gender-recognition

@REM All sercers have same "model server version" & "ip address"
SET MODEL_SERVER_VERSION="latest"
SET IP_ADDRESS=%1

setlocal ENABLEDELAYEDEXPANSION

@REM All servers are going to be started
for /l %%i in (0,1,6) do (
    SET MODEL_PATH="az://ovms/!models[%%i]!"
    SET AZURE_STORAGE_CONNECTION_STRING="AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;DefaultEndpointsProtocol=http;BlobEndpoint=http://!IP_ADDRESS!:10000/devstoreaccount1;QueueEndpoint=http://!IP_ADDRESS!:10001/devstoreaccount1;TableEndpoint=http://!IP_ADDRESS!:10002/devstoreaccount1;"
    docker run --rm -d -v !dir!:/log -p !PORT_NUMBER!:9000 -e AZURE_STORAGE_CONNECTION_STRING=!AZURE_STORAGE_CONNECTION_STRING! openvino/model_server:!MODEL_SERVER_VERSION! --model_path !MODEL_PATH! --model_name !models[%%i]! --port 9000 --log_level DEBUG --log_path "/log/!models[%%i]!.log" --file_system_poll_wait_seconds 0
)
