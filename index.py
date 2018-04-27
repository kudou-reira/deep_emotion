from flask import Flask, abort, request, jsonify
from flask_cors import CORS, cross_origin
from utils import resize_flatten_image, stringToImage, toRGB, toGray, predict, evaluate, sample
import numpy as np
import json
import cv2


prototxt = './caffe/deploy.prototxt.txt'
model = './caffe/res10_300x300_ssd_iter_140000.caffemodel'


session = { 'counter': 0 }
fileDir = "/test"

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/processImage', methods=['POST'], )
@cross_origin(supports_credentials=True)
def main():
    # print(request.data)

    print(type(request.data))
    received = request.data.decode("utf-8")
    data = json.loads(received)

    img = stringToImage(data["img"])
    img = toRGB(img)

    # create a copy for return
    img2 = img

    (h, w) = img.shape[:2]
    print(img.shape)

    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    dim_x_start = 0
    dim_y_start = 0
    dim_x_end = 0
    dim_y_end = 0

    for i in range(0, detections.shape[2]):
        # get the probability of face or not
        confidence = detections[0, 0, i, 2]

        # filter out detections based on "confidence"
        if confidence > 0.8:
            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # bounding box of the face
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10


            cv2.rectangle(img2, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(img2, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            dim_x_start, dim_y_start, dim_x_end, dim_y_end = startX, startY, endX, endY

    print(dim_x_start, dim_y_start, dim_x_end, dim_y_end)
    crop_img = img[dim_y_start:dim_y_end, dim_x_start:dim_x_end, :]
    crop_img = toGray(crop_img)
    print(crop_img)
    crop_img = resize_flatten_image(crop_img, 48)
    sample_img = sample(img2)

    # save_str = "/" + "test/" + "test" + str(session["counter"]) + '.jpg'
    # print(save_str)
    # cv2.imwrite("/" + "test" + "test" + str(session["counter"]) + '.jpg', crop_img)
    # cv2.waitKey(1)
    # session["counter"] = session["counter"] + 1

    # print(crop_img)
    # print(crop_img.shape)

    result = predict(crop_img)
    emotion_probability, emotion_result = evaluate(result)

    print(result)
    print("emotion result", emotion_result)
    print("emotion_probability", emotion_probability)

    return jsonify({
        'emotion_probability': emotion_probability,
        'emotion_result': emotion_result,
        'sample_image': sample_img
    })


@app.route('/test')
def index():
    return "this is the test page"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
