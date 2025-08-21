import Start as s
import json

class Clas:
    rawText = ""
    tokens = []
    word_frequency = ""
    perplexity = 0
    burstiness_scores = ""

    def setText(self, text):
        self.rawText = text

    def getText(self):
        return self.rawText
    
    def setTokens(self):
        self.tokens = s.get_tokens_from_doc(self.rawText)

    def setWordFrequency(self):
        self.word_frequency = s.get_word_frequency(self.tokens)

    def setPerplexity(self):
        self.perplexity = s.get_perplexity(self.rawText)

    def calculateBurstiness(self):
        self.burstiness_scores = s.calculate_burstiness(self.tokens)

    def getAll(self):
        dictionary = {"Text": self.rawText, "tokens": self.tokens,
                      "word_frequency": self.word_frequency, "perplexity": self.perplexity,
                      "burstiness_scores": self.burstiness_scores}
        textInJson = json.dumps(dictionary)
        return textInJson
