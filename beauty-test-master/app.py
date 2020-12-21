from __future__ import division, print_function
# coding=utf-8
import os
import sys
import glob
import re
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
import face_recognition as fr


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model/beauty_test_model.h5'

model=load_model(MODEL_PATH)
model.summary()
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img_path, model):

    image = fr.load_image_file(img_path)
    encs = fr.face_encodings(image)
    preds = model.predict(np.array(encs))
    print(type(preds))
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        print(preds[0])
        t=round(preds[0][0]*20,3)
        print('t:',t)
        print(type(t))

        result=str(t)
        print(result)
        result1 = str(result)
        return result1
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()


