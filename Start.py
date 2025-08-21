import math

import spacy
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


def get_tokens_from_text(text: str):
    return get_tokens_from_doc(nlp(text))


# This will get the frequency of which individual words apear in the text
def get_word_frequency(tokens):
    return Counter(tokens)


# This function will do the perplexity calculation based on the function:
# PP(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i)}
def get_perplexity(text):
    n_grams = list(text.noun_chunks)

    n_gram_quantity = len(n_grams)

    frequency_distribution = get_word_frequency(get_tokens_from_doc(text))

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
    sia = SentimentIntensityAnalyzer()

    return sia.polarity_scores(doc.text)


def extract_named_entities(text):
    # As I'm using both nltk and spacy, to avoid conflicts it is better to keep this way
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    named_entities = ne_chunk(pos_tags)
    return named_entities


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


def analyse_readability(doc):
    reading_ease = textstat.flesch_reading_ease(doc.text)
    kincaid_grade = textstat.flesch_kincaid_grade(doc.text)
    gunning_fog = textstat.gunning_fog(doc.text)

    return {
        "Flesch Test Reading Ease": reading_ease,
        "Flesch-Kincaid Test Grade": kincaid_grade,
        "Gunning Fog Index": gunning_fog
    }


doc = nlp(open("Sample.txt").read())
tokens = get_tokens_from_doc(doc)
sentences = [word_tokenize(sentence) for sentence in nltk.sent_tokenize(doc.text)] # Also consequence of spacy and nltk conflict

frequency = get_word_frequency(tokens)

perplexity = get_perplexity(doc)

burstiness = calculate_burstiness(tokens)

sentiment = sentiment_analyser(doc)

named_entities = extract_named_entities(doc.text)

coherence = analyse_coherence(sentences)

readability = analyse_readability(doc)

print(f"Example being Used: \n{doc} \n")
print(f"Tokenized Text: \n{tokens} \n")
print(f"Words Frequency: \n{frequency}\n")
print(f"Perplexity: \n{perplexity}\n")
print(f"Burstiness of example: \n{burstiness}\n")
print(f"Sentiment: \n{sentiment}\n")
print(f"Named Entities: \n{named_entities}\n")
print(f"Coherence Between Sentences: \n{coherence}\n")
print(f"Readability Scores: \n{readability}\n")
