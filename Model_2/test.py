from collections import OrderedDict

import nltk
import spacy

from pre_processing import to_process

nlp = spacy.load('en')

doc = nlp(u"I don't like those books.")

for token in doc:
    print(token.lemma_)

texts = ["i can't do it", "i really loved those games", "i'm not the happiest guy",
         "it's not really good"]

texts = to_process(texts, '6', 0)

print(texts)