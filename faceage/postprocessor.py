from ..shared_code import preprocessor as prep
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import cv2
import grpc
import cv2
import os
import numpy as np

_HOST = os.environ.get("AGEGENDER_IPADDRESS")
_PORT2 = os.environ.get("AGEGENDER_PORT")

# draw something
def draw_data(frame, faces, data, age):
    for a in range(faces):

        # draw rectangle---------------------------------------
          # thickness of rectangle
        thickness_rec=int((data[a][2]-data[a][0])/50)
        cv2.rectangle(frame,
                      tuple(data[a][0:2]),
                      tuple(data[a][2:4]),
                      (0,255,0),
                      thickness_rec)

        # measure text length of age information---------------
          # size of text
        size=(data[a][2]-data[a][0])/100
        length=cv2.getTextSize(f'Age:{age[a]}',
                                cv2.FONT_HERSHEY_SIMPLEX,
                                size,
                                1)[1]

        # draw text -------------------------------------------
          # These parameters (x, y) are coordinates and
          # basic point of text(age information)
        x=data[a][0]+int((data[a][2]-data[a][0])/2)-length*5
        y=data[a][3]-(data[a][3]-data[a][1])-length
        thickness_text=int(length/5)
        cv2.putText(frame,
                    f'Age:{age[a]}',
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    size,
                    (0, 255, 0),
                    thickness_text)
    return frame

# recognize age
def recognize_age(request_age, frame):

    frame=cv2.resize(frame, (62, 62))
    frame = frame.astype(np.float32)
    # hwc > bchw [1,3,62,62]
    frame = prep.transpose(frame)
    request_age.inputs["data"].CopyFrom(make_tensor_proto(frame))
    # send to infer model by grpc
    channel = grpc.insecure_channel("{}:{}".format(_HOST, _PORT2))
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    result = stub.Predict(request_age, timeout=10.0)
    outputs = make_ndarray(result.outputs['age_conv3'])
    age = outputs[0][0][0][0] * 100
    
    return int(age)

def finding_faces(frame, out_face):

    faces, data, age=0, [], []

    # age detection
    request_age = predict_pb2.PredictRequest()
    request_age.model_spec.name = 'age-gender-recognition'

    # Process all detected face regions one by one.
    for detection in out_face:

        # get confidence value
        confidence = float(detection[2])

        # convert bounding box coordintes into "frame" image scale
        x_1, y_1, x_2, y_2 = int(detection[3] * frame.shape[1]), int(detection[4] * frame.shape[0]), \
                             int(detection[5] * frame.shape[1]), int(detection[6] * frame.shape[0])
        
        # if confidence value is over 0.5, that is a "face"
        if confidence > 0.5:
            
            #clip face coordinates
            x_1, y_1 = np.clip([x_1, y_1], 0, None)
            x_2, y_2 = np.clip([x_2], None, frame.shape[1])[0], np.clip([y_2], None, frame.shape[0])[0]
            
            #append face coordinates
            data.append([x_1, y_1, x_2, y_2])
            #add face numbers
            faces+=1

            age_temp=recognize_age(request_age, frame[ y_1: y_2, x_1: x_2 ])
            # append age information
            age.append(age_temp)


    # draw rectangles and age on them
    frame=draw_data(frame, faces, data, age)

    return frame