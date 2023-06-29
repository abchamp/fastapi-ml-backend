
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
# from tensorflow.keras.models import load_model
import base64
import numpy as np
import cv2

app = FastAPI()


model_path = "./models/model_at_50.h5"
model = tf.keras.models.load_model(model_path)


class ImageRequestBody(BaseModel):
    image: str


@app.get("/")
def app_health():
    return {"status": "OK"}


@app.post("/predict")
def model_prediction(reqBody: ImageRequestBody):
    imBytes = base64.b64decode(reqBody.image)
    # im_arr is one-dim Numpy array
    imgArr = np.frombuffer(imBytes, dtype=np.uint8)
    realImage = cv2.imdecode(imgArr, flags=cv2.IMREAD_COLOR)
    # cv2.imwrite("./test.png", img)
    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgbTensor = tf.convert_to_tensor(realImage, dtype=tf.float32)
    # Add dims to rgb_tensor
    rgbTensor = tf.expand_dims(rgbTensor, 0)
    rgbTensor = tf.image.resize(rgbTensor, (200, 200))

    probabilityModel = tf.keras.Sequential([model,
                                            tf.keras.layers.Softmax()])
    predictions = probabilityModel.predict(rgbTensor, steps=1)
    print(predictions)
    return {"status": "OK"}
