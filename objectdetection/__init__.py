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
from PIL import Image
import io


def to_pil_image(img_bin):
    _decoded = io.BytesIO(img_bin)
    return Image.open(_decoded)

def resize(img: Image, w=256, h=256):
    img = img.resize((w, h), Image.ANTIALIAS)
    # TODO scale+padding is better
    return img

def transpose(data: np.array):
    # hwc > bchw
    new_shape = (2, 0, 1)
    r = data.transpose(new_shape)
    r = np.expand_dims(r, axis=0)
    return r

_HOST = os.environ.get("OBJECTDETECTION_IPADDRESS")
_PORT = os.environ.get("OBJECTDETECTION_PORT")


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

            # pre processing
            # get image_bin form request
            img_bin = files.read()
            img = to_pil_image(img_bin)
            img_cv_copied = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            # w,h = 256,256
            img = resize(img)
            img_np = np.array(img)
            img_np = img_np.astype(np.float32)
            # hwc > bchw [1,3,256,256]
            img_np = transpose(img_np)
            # print(img_np.shape)

            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'object-detection'
            request.inputs["image"].CopyFrom(
                make_tensor_proto(img_np, shape=img_np.shape))
            # send to infer model by grpc
            start = time()
            channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT))
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
            result = stub.Predict(request, timeout=10.0)
            res = make_ndarray(result.outputs["detection_out"])

            logging.warning(f'OutputType:{type(result)}')

            # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
            # where N is the number of detected bounding boxes
            detections = res.reshape(-1, 7)

            img_copied = np.copy(img_cv_copied)
            image_h, image_w = img_copied.shape[:2]
            for i, detection in enumerate(detections):

                _, class_id, confidence, xmin, ymin, xmax, ymax = detection

                if confidence > 0.5:

                    xmin = int(xmin * image_w)
                    ymin = int(ymin * image_h)
                    xmax = int(xmax * image_w)
                    ymax = int(ymax * image_h)

                    #logging.info(f'Found: label = {label}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

                    # Draw a bounding box on a output image
                    cv2.rectangle(img_copied, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            timecost = time()-start
            logging.info(f"Inference complete,Takes{timecost}")

            # post processing
            response_image = img_copied

            imgbytes = cv2.imencode('.jpg', response_image)[1].tobytes()
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
