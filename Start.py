import math

import spacy
from collections import Counter

import numpy

from transformers import pipeline

nlp = spacy.load("en_core_web_sm")


# This will get the tokens form a text
def get_tokens_from_text(doc):
    return [token.text for token in doc if token.is_stop is not True and token.is_punct is not True and token.is_space is not True]


# This will get the frequency of which individual words apear in the text
def get_word_frequency(tokens):
    return Counter(tokens)


# This function will do the perplexity calculation based on the function:
# PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i)}
def get_perplexity(text):
    n_grams = list(text.noun_chunks)

    n_gram_quantity = len(n_grams)

    frequency_distribution = get_word_frequency(get_tokens_from_text(text))

    entropy = -sum(frequency_distribution[ng] * math.log2(frequency_distribution[ng]) for ng in frequency_distribution)

    return 2 ** (entropy / n_gram_quantity)


# Calculate
def calculate_burstiness(tokens):
    word_positions = {}

    # Get the index of each time a token appears and where
    for i, word in enumerate(tokens):
        if word not in word_positions:
            word_positions[word] = []
        word_positions[word].append(i)

    # if a word has more than one appearence get the difference between positions
    # and dived the standard deviation and divide it by the mean of the deviation
    burstiness_scores = {}
    for word, positions in word_positions.items():
        if len(positions) > 1:
            gaps = numpy.diff(positions)
            burstiness = numpy.std(gaps) / numpy.mean(gaps)
            burstiness_scores[word] = burstiness

    return burstiness_scores


def sentiment_analyser(doc):

    return doc
        

doc = nlp(open("Sample.txt").read())

tokens = get_tokens_from_text(doc)

frequency = get_word_frequency(tokens)

perplexity = get_perplexity(doc)

burstiness = calculate_burstiness(tokens)

print(f"Example being Used: \n{doc} \n")
print(f"Tokenized Text: \n{doc.retokenize()} \n")
print(f"Words Frequency: \n{frequency} \n")
print(f"Perplexity: \n{perplexity}\n")
print(f"Burstiness of example: \n{burstiness} \n")
