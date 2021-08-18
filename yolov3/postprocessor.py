import cv2
import numpy as np
from math import exp as exp

# object label list
LABELS=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# parameter list
params={'anchors': [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0, 198.0, 373.0, 326.0], 'axis': 1, 'coords': 4, 'classes': 80, 'end_axis': 3, 'num': 9, 'do_softmax': False, 'mask': [3, 4, 5]}


class YoloParams_v3:
    # ------------------------------------------- extract layer parameters ------------------------------------------
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        self.side = side
        self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
                        198.0, 373.0, 326.0] if 'anchors' not in param else param['anchors']

        self.isYoloV3 = False

        if param.get('mask'):
            mask = param['mask']
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

            self.isYoloV3 = True # Weak way to determine but the only one.

class Yolo_v3:
	@staticmethod
	def entry_index(side, coord, classes, location, entry):
		side_power_2 = side ** 2
		n = location // side_power_2
		loc = location % side_power_2
		return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)

	@staticmethod
	def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
		xmin = int((x - w / 2) * w_scale)
		ymin = int((y - h / 2) * h_scale)
		xmax = int(xmin + w * w_scale)
		ymax = int(ymin + h * h_scale)
		return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)

	@staticmethod
	def intersection_over_union(box_1, box_2):
		width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
		height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
		if width_of_overlap_area < 0 or height_of_overlap_area < 0: area_of_overlap = 0
		else: area_of_overlap = width_of_overlap_area * height_of_overlap_area
		box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
		box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
		area_of_union = box_1_area + box_2_area - area_of_overlap
		if area_of_union == 0: return 0
		return area_of_overlap / area_of_union
	@staticmethod
	def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
		if is_proportional:
			scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
			offset = 0.5*(np.ones(2) - scale)
			x, y = (np.array([x, y]) - offset) / scale
			width, height = np.array([width, height]) / scale
		xmin = int((x - width / 2) * im_w)
		ymin = int((y - height / 2) * im_h)
		xmax = int(xmin + width * im_w)
		ymax = int(ymin + height * im_h)
		return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())

	@staticmethod
	def parse_yolo_region(blob, resized_image_shape, frameinal_im_shape, params, threshold):
		# ------------------------------------------ output parameters ------------------------------------------
		_, _, out_blob_h, out_blob_w = blob.shape
		assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
										"be equal to width. Current height = {}, current width = {}" \
										"".format(out_blob_h, out_blob_w)

		# ------------------------------------------ extract layer parameters -------------------------------------------
		orig_im_h, orig_im_w = frameinal_im_shape
		resized_image_h, resized_image_w = resized_image_shape
		objects = list()
		size_normalizer = (resized_image_w, resized_image_h) if params.isYoloV3 else (params.side, params.side)
		bbox_size = params.coords + 1 + params.classes
		# ------------------------------------------- analyze YOLO Region output -------------------------------------------
		for row, col, n in np.ndindex(params.side, params.side, params.num):
			# Getting raw values for each detection bounding box
			bbox = blob[0, n*bbox_size:(n+1)*bbox_size, row, col]
			x, y, width, height, object_probability = bbox[:5]
			class_probabilities = bbox[5:]
			if object_probability < threshold: continue
			# Process raw value
			x = (col + x) / params.side
			y = (row + y) / params.side
			# Value for exp is very big number in some cases so following construction is using here
			try: width, height = exp(width), exp(height)
			except OverflowError: continue
			# Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
			width = width * params.anchors[2 * n] / size_normalizer[0]
			height = height * params.anchors[2 * n + 1] / size_normalizer[1]

			class_id = np.argmax(class_probabilities)
			confidence = class_probabilities[class_id]*object_probability
			if confidence < threshold: continue
			objects.append(Yolo_v3.scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
									im_h=orig_im_h, im_w=orig_im_w, is_proportional=0.15))
		return objects

def object_detection(frame, img_np, outputs):

	COLORS = np.random.uniform(0, 255, size=(len(LABELS), 3))

	prob_threshold, iou_threshold=0.5, 0.15
	
	objects = []
	masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	
	# output object's coordinates
	for i, outBlob in enumerate(outputs):
		outBlob = outBlob.reshape(outputs[i].shape)
		params['mask']=masks[i]
		layerParams = YoloParams_v3(params, outBlob.shape[2])
		objects += Yolo_v3.parse_yolo_region(outBlob, img_np.shape[2:], \
									frame.shape[:-1], layerParams, prob_threshold)


	# measure the object's confidence
	for i in range(len(objects)):
		if objects[i]["confidence"] == 0: continue
		for j in range(i + 1, len(objects)):
			if Yolo_v3.intersection_over_union(objects[i], objects[j]) > iou_threshold:
				objects[j]["confidence"] = 0
	objects = [obj for obj in objects if obj['confidence'] >= prob_threshold]

	endY, endX = frame.shape[:-1]
	
	for obj in objects:
		# clipping
		if obj["xmax"] > endX or obj["ymax"] > endY or obj["xmin"] < 0 or obj["ymin"] < 0: continue

		label=f'{LABELS[obj["class_id"]]};{obj["confidence"] * 100:.2f}%'

		y = obj["ymin"] - 10 if obj["ymin"] - 10 > 10 else obj["ymin"] + 10

		# measure text length of age information---------------
		 # size of text
		size=(obj["xmax"]-obj["xmin"])/125
		length=cv2.getTextSize(label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                size,
                                1)[1]
		thickness_text=int(length/3.5)

		# drw a rectangle surronding objects and text which shows what it is.
		cv2.rectangle(frame, (obj["xmin"], obj["ymin"]), (obj["xmax"], obj["ymax"]), COLORS[obj["class_id"]], 2)
		cv2.putText(frame, label, (obj["xmin"], y), cv2.FONT_HERSHEY_SIMPLEX, size, COLORS[obj["class_id"]], thickness_text)

	return frame