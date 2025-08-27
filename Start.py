import math
import re

from collections import Counter

import numpy
import spacy
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize
from gensim.models import Word2Vec

nlp = spacy.load("en_core_web_sm")


# This will get the tokens form a text
def get_tokens_from_doc(doc):
    return [token.text for token in doc if
            token.is_stop is not True and token.is_punct is not True and token.is_space is not True]


def get_tokens_from_text(text):
    return get_tokens_from_doc(nlp(text))


def get_sentences(doc):
    return [word_tokenize(sentence) for sentence in nltk.sent_tokenize(doc)]


# This will get the frequency of which individual words apear in the text
def get_word_frequency(tokens):
    return Counter(tokens)


# This function will do the perplexity calculation based on the function:
# PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i)}
def get_perplexity(text):
    if isinstance(text, str):
        doc = nlp(text)
    else:
        doc = text

    n_grams = list(doc.noun_chunks)

    n_gram_quantity = len(n_grams)

    frequency_distribution = get_word_frequency(get_tokens_from_doc(doc))

    entropy = -sum(frequency_distribution[ng] * math.log2(frequency_distribution[ng]) for ng in frequency_distribution)

    return 2 ** (entropy / n_gram_quantity)


# Calculate the intermittent frequency of words in the text.
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


# Get if the feelings of the text are not very similar to machine generated text
def sentiment_analyser(doc):
    sia = SentimentIntensityAnalyzer()

    return sia.polarity_scores(doc)


# Get the entities associated with the example
def extract_named_entities(doc):
    # As I'm using both nltk and spacy, to avoid conflicts it is better to keep this way
    tokens = word_tokenize(doc)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return named_entities


# Get the text coherence between sentences
def analyse_coherence(sentences):
    # Training of the Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Get sentence embeddings
    sentence_embeddings = []
    for sentence in sentences:
        words = [word for word in sentence if word in model.wv]
        if words:
            sentence_embedding = numpy.mean([model.wv[word] for word in words], axis=0)
            sentence_embeddings.append(sentence_embedding)

    # Get similarity between consecutive sentences
    coherence_scores = []
    for i in range(len(sentence_embeddings) - 1):
        divisor = numpy.dot(sentence_embeddings[i], sentence_embeddings[i + 1])
        divident = numpy.linalg.norm(sentence_embeddings[i]) * numpy.linalg.norm(sentence_embeddings[i])

        similarity = divisor / divident

        coherence_scores.append(similarity)

    return numpy.mean(coherence_scores)


# Apply the Flesch Reading Ease, Flesch-Kincaid Test Grade and the Gunning Fog Index to the sample text
def analyse_readability(doc):
    reading_ease = textstat.flesch_reading_ease(doc)
    kincaid_grade = textstat.flesch_kincaid_grade(doc)
    gunning_fog = textstat.gunning_fog(doc)

    return {
        "Flesch Test Reading Ease": reading_ease,
        "Flesch-Kincaid Test Grade": kincaid_grade,
        "Gunning Fog Index": gunning_fog
    }


# Get the Stylometry of the text, if it is not too verbose,
# the punctuation frequency is off or the sentence length is too big
def analyse_stylometry(doc):
    sentences = nltk.sent_tokenize(doc)
    words = word_tokenize(doc)

    average_sentence_length = len(words) / len(sentences)
    verbosity_value = len(set(words)) / len(words)
    punctuation_frequency = len(re.findall(r'[^\w\s]', doc)) / len(words)

    return {
        "Average Sentence Length" : average_sentence_length,
        "Verbosity Value" : verbosity_value,
        "Punctuation Frequency" : punctuation_frequency
    }


def check_AI():
    return False
