from random import randint

import xlrd
import xlwt
from xlutils.copy import copy

from Classes.Data import Data


class Dataset:
    def __init__(self):
        self.docs = []
        self.labels = []

    def add(self, data):
        self.docs.append(data.doc)
        self.labels.append(data.label)

    @staticmethod
    def to_string(data):
        new_docs = []

        for doc in data:
            text = ""
            for word in doc:
                text = text + " " + word
            new_docs.append(text)

        return new_docs

class Data:
    def __init__(self, doc, label):
        self.doc = doc
        self.label = label

    def setlabel(self, newlabel):
        self.label = newlabel

    def tostring(self):
        return self.label, self.doc
