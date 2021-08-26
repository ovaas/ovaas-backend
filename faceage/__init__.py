from . import postprocessor as postp
from ..shared_code import preprocessor as prep
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

_HOST = os.environ.get("FACEDETECTION_IPADDRESS")
_PORT1 = os.environ.get("FACEDETECTION_PORT")
_PORT2 = os.environ.get("AGEGENDER_PORT")

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _NAME = 'image'

    event_id = context.invocation_id
    logging.info(
        f"Python face detection function start process.\nID:{event_id}\nBack-end server host: {_HOST}:{_PORT1}\n \
          Python emotion detection function start process.\nID:{event_id}\nBack-end server host: {_HOST}:{_PORT2}")

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

            # pre processing
            # get image_bin form request
            img_bin = files.read()
            img = prep.to_pil_image(img_bin)
            # rotate image with orientation value(for iOS, iPadOS)
            img=prep.rotate_image(img)
            # basic image of processing. 'frame' will be used later
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # resize image to [300, 300]
            img = prep.resize(img, w=300, h=300)
            img_np = np.array(img)
            img_np = img_np.astype(np.float32)
            # hwc > bchw [1,3,300,300]
            img_np = prep.transpose(img_np)

            # face detection
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'face-detection'
            request.inputs["input.1"].CopyFrom(make_tensor_proto(img_np))

            # send to infer model by grpc
            start = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT1))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            result = stub.Predict(request, timeout=10.0)

            # logging.warning(f'Output:{result}')
            logging.warning(f'OutputType:{type(result)}')

            pafs = make_ndarray(result.outputs['527'])
            out_face = np.squeeze(pafs)

            #-----------------------------------------------------------
            # output image which faces are surrounded with rectangles.
            # Their age are shown on them.
            frame=postp.finding_faces(frame, out_face)
            #-----------------------------------------------------------


            timecost = time()-start
            logging.info(f"Inference complete,Takes{timecost}")

            imgbytes = cv2.imencode('.jpg', frame)[1].tobytes()
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