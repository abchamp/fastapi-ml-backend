
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
# from tensorflow.keras.models import load_model
import base64
import numpy as np
import cv2

app = FastAPI()


model_path = "./app/models/model_at_50.h5"
model = tf.keras.models.load_model(model_path)


class ImageRequestBody(BaseModel):
    image: str


@app.get("/")
def app_health():
    return {"status": "OK"}


@app.post("/predict")
def model_prediction(reqBody: ImageRequestBody):
    try:
        imBytes = base64.b64decode(reqBody.image)
        # im_arr is one-dim Numpy array
        imgArr = np.frombuffer(imBytes, dtype=np.uint8)
        realImage = cv2.imdecode(imgArr, flags=cv2.IMREAD_COLOR)
        # cv2.imwrite("./test.png", img)

        imgTensor = tf.convert_to_tensor(realImage, dtype=tf.float32)
        # !!! expand_dims ??
        imgTensor = tf.expand_dims(imgTensor, 0)
        # resize image
        imgTensor = tf.image.resize(imgTensor, (200, 200))
        # probabilityModel = tf.keras.Sequential([model,
        #                                         tf.keras.layers.Softmax()])
        predictions = model.predict(imgTensor, steps=1)
        results = predictions.tolist()

        predResultLabel = ""
        if results[0][0] > .5:
            predResultLabel = 'D'
        else:
            predResultLabel = 'C'

        return {"status": "OK", "pre_label": predResultLabel, "prob": results[0][0]}
    except:
        return {"status": "!OK", "pre_label": None, results: None}
