import xlrd
import numpy as np
import matplotlib.pyplot as plt

tabela = xlrd.open_workbook('Sheets/Clustering/KNN/books.xls')

for i in range(13):
    plan = tabela.sheet_by_index(i)

    clusters = []
    for j in range(plan.ncols):
        cluster = plan.col_values(j)
        while '' in cluster:
            cluster.remove('')
        clusters.append(cluster)

    lenlist = []
    for cluster in clusters:
        lenlist.append(len(cluster))

    print(plan.name, ":")
    print("Média: ", np.mean(lenlist))
    print("Mediana: ", np.median(lenlist))
    print("Desvio padrão: ", np.std(lenlist))
    print("Mediana/desvio padrão: ", np.median(lenlist)/np.std(lenlist))
    print()
