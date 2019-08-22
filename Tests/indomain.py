import pickle

from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

nfeature = 8000
n = 500
src = 'books'
tgt = 'electronics'
maxlen = 500
batch_size = 32
filters = 250
kernel_size = 5
hidden_dims = 250
epochs = 3
nb_epoch_t = 50

# book = xlsxwriter.Workbook('Sheets/baseline1.xls')
# sheet = book.add_worksheet('1')

i = 1
for src in ['books', 'electronics', 'dvd', 'kitchen']:
    with open('Datasets/dataset_' + src, 'rb') as fp:
        dataset_source = pickle.load(fp)

    x_train, x_test, y_train, y_test = train_test_split(dataset_source.docs, dataset_source.labels,
                                                               random_state=42)

    print(len(x_train))
    print(len(x_test))

    cv = TfidfVectorizer(smooth_idf=True, norm='l1', max_features=3000)
    x_train = cv.fit_transform(x_train)
    x_test = cv.transform(x_test)

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    predict = classifier.predict(x_test)

    precision = f1_score(y_test, predict, average='binary')
    print('Precision:', precision)
    accuracy = accuracy_score(y_test, predict)
    print('Accuracy: ', accuracy)
    recall = recall_score(y_test, predict, average='binary')
    print('Recall: ', recall)
    confMatrix = confusion_matrix(y_test, predict)
    print('Confusion matrix: \n', confMatrix)
    print('\n\n\n')

    # sheet.write(i, 0, src)
    # sheet.write(0, j, tgt)
    # sheet.write(i, j, scores[1] * 100)

i += 1
# book.close()
