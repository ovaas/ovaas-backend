import cv2
import grpc
import numpy as np
import base64
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc, get_model_metadata_pb2

import configparser
import os
import io
import errno
from PIL import Image
import logging

class RemoteColorization:
    def __init__(self, grpc_address='localhost', grpc_port=9000, model_name='colorization', model_version=None):
        logging.info(f"start init")
        
        # Settings for accessing model server
        self.grpc_address = grpc_address
        self.grpc_port = grpc_port
        self.model_name = model_name
        self.model_version = model_version
        
        channel = grpc.insecure_channel("{}:{}".format(self.grpc_address, self.grpc_port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # Get input shape info from Model Server
        self.input_name, input_shape, self.output_name, output_shape = self.__get_input_name_and_shape__()
        self.input_batchsize = input_shape[0]
        self.input_channel = input_shape[1]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]
        
        # Setup coeffs
        coeffs = "./colorization/colorization-v2.npy"
        self.color_coeff = np.load(coeffs).astype(np.float32)
        assert self.color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"

    def __get_input_name_and_shape__(self):
        logging.info(f"start get_input_name")
        metadata_field = "signature_def"
        request = get_model_metadata_pb2.GetModelMetadataRequest()
        request.model_spec.name = self.model_name
        if self.model_version is not None:
            request.model_spec.version.value = self.model_version
        request.metadata_field.append(metadata_field)
        
        result = self.stub.GetModelMetadata(request, 10.0)  # result includes a dictionary with all model outputs
        input_metadata, output_metadata = self.__get_input_and_output_meta_data__(result)
        input_blob = next(iter(input_metadata.keys()))
        output_blob = next(iter(output_metadata.keys()))
        logging.info(f"get_input_name_and_shape_function success!")
        return input_blob, input_metadata[input_blob]['shape'], output_blob, output_metadata[output_blob]['shape']

    def __get_input_and_output_meta_data__(self, response):
        logging.info(f"start get_input_and_output_meta_data")
        signature_def = response.metadata['signature_def']
        signature_map = get_model_metadata_pb2.SignatureDefMap()
        signature_map.ParseFromString(signature_def.value)
        serving_default = signature_map.ListFields()[0][1]['serving_default']
        serving_inputs = serving_default.inputs
        input_blobs_keys = {key: {} for key in serving_inputs.keys()}
        tensor_shape = {key: serving_inputs[key].tensor_shape
                        for key in serving_inputs.keys()}
        for input_blob in input_blobs_keys:
            inputs_shape = [d.size for d in tensor_shape[input_blob].dim]
            tensor_dtype = serving_inputs[input_blob].dtype
            input_blobs_keys[input_blob].update({'shape': inputs_shape})
            input_blobs_keys[input_blob].update({'dtype': tensor_dtype})

        serving_outputs = serving_default.outputs
        output_blobs_keys = {key: {} for key in serving_outputs.keys()}
        tensor_shape = {key: serving_outputs[key].tensor_shape
                        for key in serving_outputs.keys()}
        for output_blob in output_blobs_keys:
            outputs_shape = [d.size for d in tensor_shape[output_blob].dim]
            tensor_dtype = serving_outputs[output_blob].dtype
            output_blobs_keys[output_blob].update({'shape': outputs_shape})
            output_blobs_keys[output_blob].update({'dtype': tensor_dtype})
        logging.info(f"Sussessed! get_input_and_output_meta_data")
        return input_blobs_keys, output_blobs_keys

    def __preprocess_input__(self, original_frame):
        if original_frame.shape[2] > 1:
            frame = cv2.cvtColor(cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l_rs = cv2.resize(img_lab.copy(), (self.input_width, self.input_height))[:, :, 0]

        return img_lab, img_l_rs

    def infer(self, original_frame):
        # Read and pre-process input image (NOTE: one image only)
        img_lab, img_l_rs = self.__preprocess_input__(original_frame)
        input_image = img_l_rs.reshape(self.input_batchsize, self.input_channel, self.input_height,
                                       self.input_width).astype(np.float32)

        # Model ServerにgRPCでアクセスしてモデルをコール
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.inputs[self.input_name].CopyFrom(
            make_tensor_proto(input_image, shape=(input_image.shape)))
        result = self.stub.Predict(request, 10.0)
        
        ##End Debug 1219 by Maiko
        res = make_ndarray(result.outputs[self.output_name])
        update_res = (res * self.color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

        out = update_res.transpose((1, 2, 0))
        (h_orig, w_orig) = original_frame.shape[:2]
        out = cv2.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2BGR), 0, 1)

        return img_bgr_out

def create_output_image(original_frame, img_bgr_out):
    logging.info(f"start create_output_image")
    # logging.info(f"img_bgr_out:{img_bgr_out.shape}")
    (h_orig, w_orig) = original_frame.shape[:2]
    logging.info(f"image height is {h_orig}, width is {w_orig}")
    im_show_size = (int(w_orig * (400 / h_orig)), 400)
    original_image = cv2.resize(original_frame, im_show_size)
    colorize_image = (cv2.resize(img_bgr_out, im_show_size) * 255).astype(np.uint8)
    colorize_image = cv2.cvtColor(colorize_image, cv2.COLOR_BGR2RGB)

    original_image = cv2.putText(original_image, 'Original', (25, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    colorize_image = cv2.putText(colorize_image, 'Colorize', (25, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    ir_image = [cv2.hconcat([original_image, colorize_image])]
    
    final_image = cv2.vconcat(ir_image)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    logging.info(f"Sussessed! create_output_image")
    return final_image

def create_input_image(files):
    image_bytes = files.read()
    # img_b64decode  = base64.b64encode(files_image)
    # img_array = np.fromstring(img_b64decode, np.uint8)
    # final_image = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
    final_image = bytesToCv2Img(image_bytes)
    return final_image


def bytesToCv2Img(bytes):
    return cv2.imdecode(np.fromstring(bytes, "uint8"), 1)
    


def cv2ImgToBytes(img):
    return cv2.imencode('.jpg', img)[1].tobytes()
