import Start as s
from flask import jsonify

class Clas:

    def __init__(self):             # Para conseguir instanciar as variáveis mais fácil
        self.rawText = ""           # Mas não se preocupe que elas estão todas como String
        self.tokens = ""            # Elas mesmas se consertam depois, mas se achar mais
        self.word_frequency = ""    # Seguro consertar aqui agora tudo bem
        self.perplexity = ""
        self.burstiness_scores = ""


    def setText(self, text):
        self.rawText = text

    def getText(self):
        return self.rawText

    # Aqui, como eu falei no Start, eu mudei o nome, porque eu não consegui
    # Fazer com que eles tivessem o mesmo nome, e diferenciar por Override    
    def setTokens(self):
        self.tokens = s.get_tokens_from_text(self.rawText)

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
        textInJson = jsonify(dictionary) # Funciona direitinho :)
        return textInJson
