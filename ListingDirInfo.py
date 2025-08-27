import Start as s
import os
import csv

path = 'Samples\\Human Text'
results_path = 'results.csv'

print("Origin | Perplexity | Burstiness | Sentiment | Coherence | Readability | Stylometry")

data = [["Origin", "Perplexity", "Burstiness", "Sentiment", "Coherence", "Readability", "Stylometry"]]

for file in os.listdir(path):
    if file.endswith('.txt'):
        file_path = os.path.join(path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tokens = s.get_tokens_from_text(content)
            sentences = s.get_sentences(content)
            print(f"{file} | {s.get_word_frequency(tokens)} | {s.get_perplexity(content)} | {s.calculate_burstiness(tokens)} | {s.sentiment_analyser(content)} | {s.analyse_coherence(sentences)} | {s.analyse_readability(content)} | {s.analyse_stylometry(content)}")
            data.append([file, s.get_word_frequency(tokens), s.get_perplexity(content), s.calculate_burstiness(tokens), s.sentiment_analyser(content), s.analyse_coherence(sentences), s.analyse_readability(content), s.analyse_stylometry(content)])
            break

# print(f"Dados: \n{data}")

with open(results_path, mode='w', encoding='utf-8', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)