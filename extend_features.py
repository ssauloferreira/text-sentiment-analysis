import pickle

import gensim.models.keyedvectors as word2vec
from gensim.models import Word2Vec

def gen_model(src):

    with open('Datasets/dataset_'+src, 'rb') as fp:
        dataset = pickle.load(fp)
    sentences = dataset.docs

    #for text in sentences:
    #    print(text)

    model = Word2Vec(sentences=sentences, size=150, window=10, min_count=2, workers=10)

    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)

    model.wv.save_word2vec_format('Embeddings_Models/model_'+src+'.bin')



def extend_features(features, vocabulary, src):
    new_features = []

    path = 'Embeddings_Models/model_'+src+'.bin'

    model = word2vec.KeyedVectors.load_word2vec_format(path, binary=True)

    for feature in features:
        if feature in vocabulary:

            new_features.append(feature)
            if feature in model.vocab:
                similars = model.wv.most_similar(positive=feature)

                for sim in similars:
                    w = sim[0].lower()
                    if w not in features:
                        new_features.append(sim[0])

    new_features = list(dict.fromkeys(new_features))

    return new_features
