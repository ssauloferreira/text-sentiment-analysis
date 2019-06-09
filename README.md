## A Cluster of Features Approach for Cross-Domain Sentiment Classification
#### Saulo Lucas Gomes Ferreira


### 1. Escolha dos datasets
Os datasets utilizados são do conjunto Amazon Customer Review aplamente explorado em pesquisas da área de Cross-Domain Sentiment Analysis. No geral, são compostos por quatro domínios: livros, eletrônicos, cozinha e filmes; com 2000 amostras de texto cada um, com classificação binária, sendo 1000 positivas e 1000 negativas.

### 2. Pré-processamento
O processamento dos dados consiste nos seguintes passos:

1. Tokenization
2. Lemmatization
3. Negação de palavra
4. Remoção de features raras
5. Remoção de features neutras (de acordo com o *sentiment value*)
6. Filtragem de POS (verbos, advérbios, susbtantivos e adjetivos)
7. Atribuição de POS (i.e. book = book_n)

### 3. Representação por SentiWordNet
A biblioteca SentiWordNet, complemento da WordNet, foi utilizada para a representação do *sentiment value* de cada feature resultante do pré-processamento. Para cada palavra passada para a função, é retornada um conjunto de scores, um para cada *synset* da palavra.
Um *synset* é uma representação de uma palavra em um determinado contexto, tendo em vista que a mesma palavra pode ser utilizada em diferentes qualificações semânticas em uma sentença. Como não se sabe qual o *synset* que representa aquela palavra no contexto que ela está inserida no texto pré-processado, foram filtrados apenas os *synsets* do mesmo POS da feature. Por exemplo:

```python
def get_sentiwordnet_score(feature='book_n')
      word = ''
      pos = ''

      for i in range(len(feature)):
          if item[i] == '_':
              word = item[:i]
              pos = item[i+1:]
```
A função separa a palavra do POS, e pesquisa somente os *synsets* com aquela classe gramatical. Em seguida, lista os scores retornados para a feature correspondente. O resultado, se for mais de um, é somado em um array:
```python
      syns = list(swn.senti_synsets(word))
      if syns.__len__() > 0:
          pos_score = []
          neg_score = []
          obj_score = []
          for syn in syns:
              if pos in syn.synset.name():
                  pos_score.append(syn.pos_score())
                  neg_score.append(syn.neg_score())
                  obj_score.append(syn.obj_score())

          if len(pos_score) > 0:
              aux = [round(sum(pos_score), 3),
                     round(sum(neg_score), 3),
                     round(sum(obj_score), 3)]
              return aux
```
Com isso para cada feature do vocabulário haverá um vetor de três elementos, onde representa o *pos_score*, *neg_score* e *obj_score*. Como no exemplo seguinte:

| feature | *sentiment_value* |
|---------|-------------------|
|recomend|[0.7, 0.0, 0.3]|
|sad|[0.1, 0.8, 0.1]|
|great|[0.9, 0.0, 0.1]|

### 4. Clustering
Na fase de clustering foi utilizado o K-Means para agrupar as features em conjuntos de sentimentos semelhantes. O tamanho de clusters varia de 10 a 500, onde ainda estão sendo testados para cada algoritmo de classificação.

### 5. Ranquear as features em comum
Para critério de se obter a semelhança entre os clusters, é necessário se obter as melhores features que os domínios compartilham. Para isso foi realizado o algoritmo de *chi-square* para ranquar as features em comum, tomando como base as labels do domínio de origem. Após isso, já podemos ir para o próximo passo.

### 6. Ligar clusters
Com um conjunto de clusters de domínio origem na mão e um conjunto de clusters de domínio destino em outra, o próximo passo é mapear os clusters de um domínio no outro. Para isso é realizado o seguinte passo-a-passo.

```
Fs = melhores features em comum
Cs = clusters do domínio de origem
Ct = clusters do domínio destino
Linked = lista para armazenar indíces dos clusters agrupados

para cada cluster c1 de Cs fazer:
  FCs <- conjunto de Fs presentes em c

  para cada cluster c2 de Ct fazer:
    n = quantidade de FCs presentes em c2

  IndiceCluster = índice do cluster c1
  IndiceResultante = índice do cluster com maior n
  Linked->adiciona(IndiceCluster, IndiceResultante)
```

Com isso, teremos uma lista onde cada posição terá uma tupla contendo índices de dois clusters com maior quantidade de features ranqueadas em comum.

### 6. Ligar features
Com os clusters conectados, o objetivo seguinte é associar features correspondentes, para isso, é necessário calcular a similaridade de importância delas no texto. O objetivo aqui não é conectar sinônimos, é importante ressaltar isso, o objetivo é conectar features de pesos similares no contexto onde estão inseridas. Esses pesos são conectados utilizando o * **A**verage **L**exical **Sen**tiWordnet **T**fIdf**, *ALSENT*, definido por:

- *Average TF* = média da frequência de uma feature nos documentos onde está presente.
- *Sentiment value* = *pos_score* - *neg_score*.
- *IDF* = inverso da frequência nos documentos.

*ALSENT = Average TF x Sentiment Value x IDF*  

### 7. Transferência de pesos
A técnica para conseguir formar um vocabulário abrangente para a classificação dos dois domínios foi a transferência de pesos de features exclusivas de um domínio para o outro. Para isso, as features conectadas foram *coladas* uma na outra de uma forma que uma pudesse transferir seu peso obtido na classificação. Então, para cada duas features conectadas *fs* e *ft*, toda ocorrência de *fs* no domínio origem é substituída por *fs_ft* e toda ocorrência de *ft* no domínio destino é substituída por *fs_ft*.
Assim, o número de features em comum aumenta consideravelmente, tornando mais similar o vocabulário dos domínios, dando mais elementos classificáveis ao classificador.
