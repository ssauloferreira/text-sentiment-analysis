import csv
import pickle

from pre_processing import to_process

data = 'kitchen'

with open('Datasets/dataset_'+data, 'rb') as fp:
    data_source = pickle.load(fp)

dataset = to_process(data_source.docs, '7', 0)

tf = {}
for text in dataset:
    for word in text:
        if word[0] not in tf:
            tf[word[0]] = 1
        else:
            aux = tf[word[0]]
            aux += 1
            tf[word[0]] = aux

aux = {}

for text in dataset:
    for word in text:
        if word[0] not in aux:
            info = [tf[word[0]], word[1]]
            aux[word[0]] = info

result = []
for item in aux:
    result.append([item, aux[item][0], aux[item][1]])

csv_file = open(data+'.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerows(result)
csv_file.close()
