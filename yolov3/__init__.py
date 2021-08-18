from . import postprocessor as postp
from ..shared_code  import preprocessor as prep
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from time import time
import azure.functions as func
import cv2
import grpc
import logging
import numpy as np
import os

_HOST = os.environ.get("YOLOV3_IPADDRESS")
_PORT = os.environ.get("YOLOV3_PORT")


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _NAME = 'image'

    event_id = context.invocation_id
    logging.info(
        f"Python humanpose function start process.\nID:{event_id}\nBack-end server host: {_HOST}:{_PORT}")

    try:
        method = req.method
        url = req.url
        files = req.files[_NAME]

        if method != 'POST':
            logging.warning(
                f'ID:{event_id},the method was {files.content_type}.refused.')
            return func.HttpResponse(f'only accept POST method', status_code=400)

        if files:
            if files.content_type != 'image/jpeg':
                logging.warning(
                    f'ID:{event_id},the file type was {files.content_type}.refused.')
                return func.HttpResponse(f'only accept jpeg images', status_code=400)

            img_bin = files.read()

            img = prep.to_pil_image(img_bin)
            img=prep.rotate_image(img)

            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            # resize image to [416, 416]
            img = prep.resize(img, w=416, h=416)
            # img = prep.resize(img, w=608, h=608)
            img_np = np.array(img)
            img_np = img_np.astype(np.float32)
            # hwc > bchw [1,3,416, 416]
            img_np = prep.transpose(img_np)

            # semantic segmentation
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'yolo-v3'
            # request.model_spec.name = 'yolo-v4'
            request.inputs["input_1"].CopyFrom(make_tensor_proto(img_np))
            # request.inputs["image_input"].CopyFrom(make_tensor_proto(img_np))

            # send to infer model by grpc
            start = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            result = stub.Predict(request, timeout=10.0)

            # logging.warning(f'Output:{result}')
            logging.warning(f'OutputType:{type(result)}')

            # print(result.outputs)
            
            output1=make_ndarray(result.outputs['conv2d_58/Conv2D/YoloRegion'])
            output2=make_ndarray(result.outputs['conv2d_66/Conv2D/YoloRegion'])
            output3=make_ndarray(result.outputs['conv2d_74/Conv2D/YoloRegion'])
            outputs=[output1, output2, output3]
            #-----------------------------------------------------------
            # output image which objects are surrounded with rectangles.
            # Their labels are shown on them.
            frame=postp.object_detection(frame, img_np, outputs)
            #-----------------------------------------------------------


            timecost = time()-start
            logging.info(f"Inference complete,Takes{timecost}")

            imgbytes = cv2.imencode('.jpg', frame)[1].tobytes()
            # imgbytes = prep.encode(image)
            MIMETYPE = 'image/jpeg'

            return func.HttpResponse(body=imgbytes, status_code=200, mimetype=MIMETYPE, charset='utf-8')

        else:
            logging.warning(f'ID:{event_id},Failed to get image,down.')
            return func.HttpResponse(f'no image files', status_code=400)

    except grpc.RpcError as e:
        status_code = e.code()
        if "DEADLINE_EXCEEDED" in status_code.name:
            logging.error(e)
            return func.HttpResponse(f'the grpc request timeout', status_code=408)
        else:
            logging.error(f"grpcError:{e}")
            return func.HttpResponse(f'Failed to get grpcResponse', status_code=500)

    except Exception as e:
        logging.error(f"Error:{e}\n\
                        url:{url}\n\
                        method:{method}\n")
        return func.HttpResponse(f'Service Error.check the log.', status_code=500)