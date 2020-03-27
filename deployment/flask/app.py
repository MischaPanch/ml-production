#!/usr/bin/env python3

import logging
import os
import pickle
from json.decoder import JSONDecodeError

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import pandas as pd
from flask import Flask, request, jsonify, g
from flask.logging import default_handler
from flask_restplus import Api, Resource, fields
from sensai.vector_model import VectorModel
from werkzeug.exceptions import BadRequest

# COLLECT ENV VARIABLES ###############
LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
PORT = int(os.environ.get('PORT', 20001))
HOST = os.environ.get('HOST', None)
#######################################

logging.basicConfig(level=LOGLEVEL)
root = logging.getLogger()
root.addHandler(default_handler)
jsonpickle_numpy.register_handlers()

app = Flask(__name__)
api = Api(app)

DATA_TOKEN = 'data'

RESOURCE_FIELDS = api.model('Resource', {
    DATA_TOKEN: fields.String(description='Pandas dataframe that was encoded with jsonpickle', required=True)
})


def loadModel() -> VectorModel:
    with open("reviewClassifier-v1.pickle", 'rb') as f:
        model = pickle.load(f)
    return model


def get_model() -> VectorModel:
    if 'model' not in g:
        g.model = loadModel()

    return g.model


@api.route('/api/v1/features', methods=['post'])
class SamplePredictor(Resource):
    @api.expect(RESOURCE_FIELDS, validate=True)
    def post(self):
        content = request.get_json()
        model = get_model()
        x = content[DATA_TOKEN]
        try:
            x = jsonpickle.decode(x)  # expects a numpy array with dimensions (h, w, n_channels)
        except JSONDecodeError:
            raise BadRequest("The input string could not be decoded")
        if not isinstance(x, pd.DataFrame):
            raise BadRequest("The input has to be a numpy array. Instead got {}".format(x.__class__))

        x = model.predict(x)
        x = jsonpickle.encode(x)
        return jsonify(prediction=x)


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)

