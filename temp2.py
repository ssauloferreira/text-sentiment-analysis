import pickle
import numpy as np
from sklearn.decomposition import PCA
from math import pow
from sklearn.feature_extraction.text import CountVectorizer
from pre_processing import to_process, get_senti_representation, get_vocabulary
from sklearn.feature_selection import chi2

def to_string(lists):
    new_docs = []

    for item in lists:
        text = ""
        for word in item:
            text = text + " " + word
        new_docs.append(text)

    return new_docs

with open('dictionary', 'rb') as fp:
    grouped = pickle.load(fp)

with open('Datasets/dataset_kitchen', 'rb') as fp:
    dataset = pickle.load(fp)
data_src = to_process(dataset.docs, '1011', 3)
label_source = dataset.labels

with open('Datasets/dataset_electronics', 'rb') as fp:
    dataset = pickle.load(fp)
data_tgt = to_process(dataset.docs, '1011', 3)

vocabulary_source = get_vocabulary(data_src)
vocab_source, scores_source, dicti_source = get_senti_representation(vocabulary_source, True)

vocabulary_target = get_vocabulary(data_tgt)
vocab_target, scores_target, dicti_target = get_senti_representation(vocabulary_target, True)

cv_source = CountVectorizer(max_df=0.95, min_df=2)
x_source = cv_source.fit_transform(to_string(data_src))

chi_stats, p_vals = chi2(x_source, label_source)
chi_res = sorted(list(zip(cv_source.get_feature_names(), chi_stats)),
                    key=lambda x: x[1], reverse=True)[0:1000]
features = []
for chi in chi_res:
    features.append(chi[0])

best20grouped = []
best20ungrouped = []

i = 0
while len(best20grouped) < 10:
    if features[i] in grouped:
        best20grouped.append(features[i])
    i += 1

i = 0
while len(best20ungrouped) < 10:
    if features[i] not in grouped:
        best20ungrouped.append(features[i])
    i += 1

print(best20grouped)
print(best20ungrouped)