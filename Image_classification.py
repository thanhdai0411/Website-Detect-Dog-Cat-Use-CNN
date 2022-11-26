from __future__ import division, print_function

import os
# coding=utf-8
import sys

import numpy as np
# Flask utils
from flask import Flask, redirect, render_template, request, url_for
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model_path = 'Cat_Dog_Model.h5'
model = load_model(model_path)
# model._make_predict_function()  # Necessory


def predictions(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)

    # pred = np.argmax(model.predict(x)[0], axis=-1)
    pred = (model.predict(x)[0][0])
    print('>>>>> pred: ' + str(model.predict(x)[0][0]))

    if pred > 0.5:
        preds = "Chó"
    else:
        preds = "Mèo"

    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        img = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(img.filename))
        img.save(file_path)

        result = predictions(file_path, model)  # return index

        return result
    return None


app.run(debug=True)
