from shared_code import preprocessor as prep
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

_HOST = os.environ.get("SUPERRESOLUTION_IPADDRESS")
_PORT = os.environ.get("SUPERRESOLUTION_PORT")


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _NAME = 'image'

    event_id = context.invocation_id
    logging.info(
        f"Python super resolution function start process.\nID:{event_id}\nBack-end server host: {_HOST}:{_PORT}")

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
            img = prep.rotate_image(img)

            w, h = img.size
            h_half=int(h/2)
            w_half=int(w/2)

            # crop the image
            imgs=[img.crop((0,0,w_half,h_half)),
                  img.crop((w_half,0,w,h_half)),
                  img.crop((0,h_half,w_half,h)),
                  img.crop((w_half,h_half,w,h))]

            outputs: np.ndarray = []

            for img in imgs:
                # resize image to [640, 360]
                img0 = prep.resize(img, w=640, h=360)
                img0_np = np.array(img0)
                img0_np = img0_np.astype(np.float32)
                # hwc > bchw [1,3,640, 360]
                img0_np = prep.transpose(img0_np)

                # resize image to [1920, 1080]
                img1_cv=cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                img1_cv=cv2.resize(img1_cv, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                img1_np = img1_cv.astype(np.float32)
                # hwc > bchw [1,3,1920, 1080]
                img1_np = prep.transpose(img1_np)

                # super resolution
                request = predict_pb2.PredictRequest()
                request.model_spec.name = 'super-resolution'
                request.inputs["0"].CopyFrom(make_tensor_proto(img0_np))
                request.inputs["1"].CopyFrom(make_tensor_proto(img1_np))

                # send to infer model by grpc
                start = time()
                options = [('grpc.max_receive_message_length', 24883241)]
                channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT), options = options)
                stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
                result = stub.Predict(request, timeout=10.0)

                # logging.warning(f'Output:{result}')
                logging.warning(f'OutputType:{type(result)}')
                # print(result)
                output = make_ndarray(result.outputs['90'])

                #-----------------------------------------------------------
                # output image whose resolution was mede higher
                output = output.reshape(3, 1080, 1920).transpose(1, 2, 0)
                output = np.clip(output * 255, 0, 255)
                output = np.ascontiguousarray(output).astype(np.uint8)
                output = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)
                #-----------------------------------------------------------

                outputs.append(output)

                timecost = time()-start
                logging.info(f"Inference complete,Takes{timecost}")

            # merge all images
            frame1 = cv2.hconcat([outputs[0], outputs[1]])
            frame2 = cv2.hconcat([outputs[2], outputs[3]])
            frame = cv2.vconcat([frame1, frame2])

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