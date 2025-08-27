from flask import Flask, request
from Classes.Clas import Clas
from flask_cors import CORS, cross_origin

API = Flask(__name__)

cors = CORS(API)
example = Clas()


@cross_origin(CORS(API, resources={r"/api/*": {"origins": "http://localhost:3000"}}))
@API.route('/text', methods=['POST'])
def returnAll():
    jsonTest = request.get_json()
    example.setText(jsonTest['text'])

    example.setTokens()
    example.setWordFrequency()
    example.setPerplexity()
    example.calculateBurstiness()

    return example.getAll()

@API.route('/test',methods=['GET'])
def returnTest():
    return example.getText()

# @API.route('/tokens_from_text',methods=['GET'])
# def get_tokens_from_text():
#     return s.get_tokens_from_text()

# @API.route('/word_frequency',methods=['GET'])
# def get_word_frequency():
#     return s.get_word_frequency()

# @API.route('/filter_punctuation',methods=['GET'])
# def filter_punctuation():
#     return s.filter_punctuation()

# @API.route('/get_perplexity',methods=['GET'])
# def perplexity():
#     return s.perplexity()

# @API.route('/calculate_burstiness',methods=['GET'])
# def calculate_burstiness():
#     return s.calculate_burstiness()

@API.route('/text',methods=['POST'])
def post_method(text):
    new_text = text
    example.setText(new_text)
    return example.getText()

API.run(port=500, host='localhost', debug=True)