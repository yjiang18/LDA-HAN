import numpy as np

from flask import Flask
from flask import render_template
from flask import request
from flask import Response
from flask import jsonify
import tensorflow as tf
from keras import backend as K
import os
from utils.normalize import normalize
from lda_han import LDA_HAN

SAVED_MODEL_DIR = 'checkpoints'
SAVED_MODEL_FILENAME = 'lda_HAN_best_1.h5'

app = Flask(__name__)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

K.clear_session()
h = LDA_HAN()
h.load_weights(SAVED_MODEL_DIR, SAVED_MODEL_FILENAME)
graph = tf.get_default_graph()


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/activations')
def activations():
    """
	Receive a text and return HNATT activation map
	"""

    if request.method == 'GET':
        text = request.args.get('text', '')
        if len(text.strip()) == 0:
            return Response(status=400)

        ntext = normalize(text)

        global graph
        with graph.as_default():
            activation_maps = h.activation_maps(text, websafe=True)
            preds = h.predict(text)
            print(preds)
            prediction = float(preds)
            print(prediction)
            data = {
                'activations': activation_maps,
                'normalizedText': ntext,
                'prediction': prediction,
                'binary': True
            }
            return jsonify(data)
    else:
        return Response(status=501)
