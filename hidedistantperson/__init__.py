from . import postprocessor as postp
from ..shared_code import preprocessor as prep
from tensorflow import make_tensor_proto, make_ndarray, cast
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

_MONODEPTH_HOST = os.environ.get("MONODEPTH_IPADDRESS")
_MONODEPTH_PORT = os.environ.get("MONODEPTH_PORT")

_HUMANSEGMENTATION_HOST = os.environ.get("HUMANSEGMENTATION_IPADDRESS")
_HUMANSEGMENTATION_PORT = os.environ.get("HUMANSEGMENTATION_PORT")

def request_caching(img_np: Image, _HOST: str, _PORT: str, model_name: str, input_name: str, output_name: str) -> np.array:

    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.inputs[input_name].CopyFrom(make_tensor_proto(img_np))

    options = [('grpc.max_receive_message_length', 8388653)]
    channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT), options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Predict(request, timeout=10.0)

    logging.warning(f'OutputType:{type(result)}')

    output = result.outputs[output_name]
    output = make_ndarray(output)

    return output

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _NAME = 'image'

    event_id = context.invocation_id
    logging.info(
        f"Python 'hide distant person' function start process.\nID:{event_id}\n\
            Back-end server host: {_MONODEPTH_HOST}:{_MONODEPTH_PORT} & {_HUMANSEGMENTATION_HOST}:{_HUMANSEGMENTATION_PORT}")

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
            # get width and height value of img
            w, h=img.size

            # resize image to [384, 384] for monodepth model
            img_np_mono = prep.resize(img, w=384, h=384)
            img_np_mono = np.array(img_np_mono)
            img_np_mono = img_np_mono.astype(np.float32)
            # hwc > bchw [1,3,384, 384]
            img_np_mono = prep.transpose(img_np_mono)

            # resize image to [2048, 2048] for humansegmentation model
            img_np_human = prep.resize(img, w=2048, h=1024)
            img_np_human = np.array(img_np_human)
            img_np_human = img_np_human.astype(np.float32)
            # hwc > bchw [1,3,2048,2048]
            img_np_human = prep.transpose(img_np_human)


            # send to infer model by grpc
            start = time()
            output_mono=request_caching(img_np_mono, _MONODEPTH_HOST, _MONODEPTH_PORT, \
                                        'midasnet', 'image', 'inverse_depth')
            output_human=request_caching(img_np_human, _HUMANSEGMENTATION_HOST, _HUMANSEGMENTATION_PORT, \
                                         'semantic-segmentation-adas', 'data', '4656.1')

            #-----------------------------------------------------------
            # make a monodepth mask image
            output_mono = np.squeeze(output_mono)
            maskout_mono=postp.make_mono_mask(output_mono, w, h)

            # make human segmentation mask image
            maskout_human=postp.make_human_mask(output_human, w, h)
            
            # make a black image
            black = np.zeros((h, w, 3))
            black = black.astype(np.uint8)
            black = Image.fromarray(black)

            # composite mask and image with black image
            # make an image by blending 'black' &'img' using 'maskout_mono'.
            image=Image.composite(black, img, maskout_mono)

            # make an image by blending 'img' & 'image' using 'maskout_human'.
            image=Image.composite(img, image, maskout_human)

            # image=Image.composite(img, black, maskout_human)
            # image=Image.composite(image, img, maskout_mono)

            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

            #-----------------------------------------------------------


            timecost = time()-start
            logging.info(f"Inference complete,Takes{timecost}")

            imgbytes = cv2.imencode('.jpg', image)[1].tobytes()
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