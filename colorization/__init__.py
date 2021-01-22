from . import preprocessor as prep
from time import time
import azure.functions as func
import logging
import os


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    _HOST = os.environ.get("COLORIZATION_IPADDRESS")
    _PORT = os.environ.get("COLORIZATION_PORT")

    event_id = context.invocation_id
    logging.info(
        f"Python colorization function start process.\nID:{event_id}\nback server host:{_HOST}:{_PORT}")

    method = req.method
    url = req.url

    files = req.files['image']
    if method != 'POST':
        logging.warning(
            f'ID:{event_id},the method was {files.content_type}.refused.')
        return func.HttpResponse(f'only accept POST method', status_code=400)
    if not files:
        logging.warning(f'ID:{event_id},Failed to get image,down.')
        return func.HttpResponse(f'no image files', status_code=400)

    start_time = time()

    # pre processing
    input_image = prep.create_input_image(files)  # get image form request
    logging.info(f'Input_Image Success.')

    try:
        img_bgr_out = prep.RemoteColorization(_HOST, _PORT).infer(input_image)

        logging.info(f"Colorization success!")

    except Exception as e:
        if 'StatusCode.DEADLINE_EXCEEDED' in str(e):
            logging.error(e)
            return func.HttpResponse(f'The gRPC request time out', status_code=408)
        else:
            logging.error(f"Error:{e}\n\
                            url:{url}\n\
                            method:{method}\n")
            return func.HttpResponse(f'Service Error.check the log.', status_code=500)

    time_cost = time() - start_time

    logging.info(f"Inference completed.Takes {'%.1f'%time_cost} seconds.")

    final_image = prep.create_output_image(input_image, img_bgr_out)
    # logging.info(f"Successfully.final_image is {final_image}.")
    mimetype = 'image/jpeg'

    img_output_bytes = prep.cv2ImgToBytes(final_image)
    logging.info(f"Success!Return response.")
    return func.HttpResponse(body=img_output_bytes, status_code=200, mimetype=mimetype, charset='utf-8')
