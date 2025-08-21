from flask import Flask, request
from Clas import Clas

API = Flask(__name__)

example = Clas()

@API.route('/text',methods=['GET'])
def returnAll():
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
def post_method():
    new_text = request.get_data(as_text=True)
    example.setText(new_text)
    return example.getText()

API.run(port=500, host='localhost', debug=True)