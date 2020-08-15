#!/usr/bin/python

#bibliotecas
import sys
import pickle
import matplotlib.pyplot
import re
import os
import pandas as pd

#funcoes
sys.path.append("../tools/")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score


#funcoes da udacity
from parse_out_email_text import parseOutText
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


#Caso a pasta maildir nao exista na maquina, coloque as linhas de codigo do
#leia-me embaixo dessa obeservacao

import urllib
url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
urllib.urlretrieve(url, filename="../enron_mail_20150507.tar.gz")
print "download complete!"


print
print "unzipping Enron dataset (this may take a while)"
import tarfile
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

### Task 1: Select what features you'll use.
features_list = ['poi','salary', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remover outliers
data = featureFormat(data_dict, features_list)
#verificando outilers por meio visual.
"""
print "Vizualizacao: salary e total_stock_value"
for point in data:
    salary = point[1]
    total_stock_value = point[2]
    matplotlib.pyplot.scatter( salary, total_stock_value )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()
"""

data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

#explorando o dataset da Enron
print "\n\n#### INFOS DO DATASET ####\n"
print "Numero de observacaoes: ", len(data_dict.keys())
poi=0
n_poi=0
salary=0
tsv = 0
for name in data_dict.keys():
    if data_dict[name]["poi"]==1:
        poi = poi+1
    else:
        n_poi=n_poi+1

    if data_dict[name]["salary"]!="NaN":
        salary = salary+1

    if data_dict[name]["total_stock_value"]!="NaN":
        tsv = tsv+1

print "Numero de pois: ",poi
print "Numero de nao pois: ",n_poi
print "Numero de NaN`s nos atributos de salario: ",data.shape[0]-salary
print "Numero de NaN`s nos atributos de total_stock_value: ", data.shape[0]-tsv

### Task 3: Create new feature(s)
print "\n\nCriando atributos..."
#Criando atributos para_poi e de_poi
for name in data_dict:
    data_point = data_dict[name]
    para = 0
    de = 0
    if (data_point['from_this_person_to_poi']!= 'NaN') & (data_point['from_messages']!= 'NaN'):
        para = data_point['from_this_person_to_poi']/data_point['from_messages']
    if (data_point['from_poi_to_this_person']!= 'NaN') & (data_point['to_messages']!= 'NaN'):
        de = data_point['from_poi_to_this_person']/data_point['to_messages']
    data_point["de_poi"] = de
    data_point["para_poi"] = para
    data_dict[name] = data_point
print "Atributos de_poi e para_poi criados!"
#criando atribo alavancagem
for name in data_dict:
    data_point = data_dict[name]
    alavanca = 0
    if (data_point['salary']!= 'NaN') & (data_point['total_stock_value']!= 'NaN'):
        alavanca = data_point['total_stock_value']/data_point['salary']
    data_point["alavancagem"] = alavanca
    data_dict[name] = data_point
print "Atributo alavancagem criado!"
#Criarei atributos de palavras com base nos emails da Enron
word_data = []
from_data = []
p = []
#ler emails de cada colaborador que possui um endereco de email no dicionario
#nesse processo a funcao parseOutText ja retira os stopwords
print "Iniciando leitura dos emails"
for name in data_dict:
    try:
        from_person = open("emails_by_address/from_" + data_dict[name]["email_address"] + ".txt", "r")
        temp_counter = 0
        words = ""
        for path in from_person:
            #por questao de tempo de processamento, so li 100 emails de cada pessoa
            if temp_counter < 100:
                path = os.path.join('..', path[20:len(path)-1:])
                email = open(path, "r")
                w = parseOutText(email)
                words = words + w
                email.close()
            temp_counter += 1
        word_data.append(words)
        from_data.append(name)
        if data_dict[name]["poi"]==1:
            p.append(1)
        else:
            p.append(0)
        from_person.close()
    except:
        continue
print "emails processados"

#transformar a lista de emails de cada colaborador em um vetor de palavras
vectorizer = TfidfVectorizer(sublinear_tf=True, use_idf=False, max_df=0.5,stop_words= 'english')
vec = vectorizer.fit_transform(word_data)
#selecionar as palavras mais significantes. Isso reduz a quantidade de atributos
#que irei colocar no dicionario
print "Escolhendo palavras mais significativas..."
selector = SelectPercentile(f_classif, percentile=5).fit(vec, p)

#matriz de consulta do peso das palavras.
word_freq = vec.toarray()
palavras = vectorizer.get_feature_names()

#faz uma listas das palavras que eu quero colocar no dicionario
word_selected = []
for word in palavras:
    if selector.get_support()[palavras.index(word)]==1:
        word_selected.append(word)
print "Tamanho da lista de palavras mais significaticas: ", len(word_selected)
print "Tamanho do dicionario antes de incluir as palavras", len(data_dict["LAY KENNETH L"].keys())

#coloca no dicionario
indx = 0
for name in data_dict:
    data_point = data_dict[name]
    #verifica se o nome atual do dict e um nome no from_data
    if name == from_data[indx]:
        for word in word_selected:
            data_point[word]= word_freq[indx][palavras.index(word)]
            data_dict [name]= data_point
        indx = indx +1
    else:
        for word in word_selected:
            data_point[word] = 'NaN'
            data_dict[name]= data_point
print "Tamanho do dicionario depois de incluir palavras", len(data_dict["LAY KENNETH L"].keys())

### Store to my_dataset for easy export below.
my_dataset = data_dict
#escalonadno caracteristicas
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data)
scaled_features = zip(my_dataset.keys(),scaled_features)
for name, feature in  (scaled_features):
    data_point = my_dataset[name]
    data_point["salary"] = feature[1]
    data_point["total_stock_value"] =feature[2]
    my_dataset[name] = data_point
print "Caracteristicas escalonadas!"

"""
print "\nTestando classificador sem adicao de novas features:"
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2, random_state=42, shuffle = True)

clf3 = DecisionTreeClassifier(min_samples_split=20, random_state=0)
clf3.fit(features_train,labels_train)
pred3 = clf3.predict(features_test)
print "Acuracia: ", accuracy_score(labels_test, pred3,normalize = True)
print "Precision: ",precision_score(labels_test, pred3)
print "Recall: ", precision_score(labels_test, pred3)
print "F1_Score: ", f1_score(labels_test, pred3)
"""
print "\n Adicionando caracteristicas na analise"
#juntando as caracteristicas iniciais escalonadas e as criadas
features_list = features_list + word_selected +['alavancagem','de_poi','para_poi']
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.2, random_state=42, shuffle = True)

"""
***Retirei essa parte do codigo para deixar apenas o classificador final***

###aqui eu testei o classificador com diferentes quantidade de caracteristcias###

print "\nResultados do classificador com as..."
for i in [50,100,200,300,400,500,600,700,800,900,1000]:
    best = SelectKBest(chi2, k=i).fit(features_train, labels_train)
    features_train_best = best.transform(features_train)
    features_test_best = best.transform(features_test)
    clf3 = DecisionTreeClassifier(min_samples_split=2,random_state=0)
    clf3.fit(features_train_best,labels_train)
    pred3 = clf3.predict(features_test_best)
    print i,"melhores features: ", f1_score(labels_test, pred3)

print "\n\nIniciando o teste de classificadores"

clf1 = GaussianNB()
clf1.fit(features_train, labels_train)
pred1 = clf1.predict(features_test)
accuracy1 = accuracy_score(labels_test, pred1,normalize = True)
precision1 = precision_score(labels_test, pred1)
recall1 = recall_score(labels_test, pred1)
print "GaussianNB finalizado"

#parameters = {'kernel':('rbf','poly'), 'C':[100]}
#svr = SVC()
#clf2 = GridSearchCV(svr, parameters)
clf2 = SVC(kernel='rbf', C=100)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
accuracy2 = accuracy_score(labels_test, pred2,normalize = True)
precision2 = precision_score(labels_test, pred2)
recall2 = recall_score(labels_test, pred2)
print "SVC finalizado"

parameters = {'n_estimators':[20,50,70],'learning_rate':[1,1.5,2]}
svr = AdaBoostClassifier()
clf4 = GridSearchCV(svr, parameters)
clf4.fit(features_train, labels_train)
pred4 = clf4.predict(features_test)
accuracy4 = accuracy_score(labels_test, pred4,normalize = True)
precision4 = precision_score(labels_test, pred4)
recall4 = recall_score(labels_test, pred4)

print "Resultados do GaussianNB: "
print "Acuracia: ", accuracy1
print "Precision: ",precision1
print "Recall: ",recall1

print "Resultados do SVC: "
print "Acuracia: ", accuracy2
print "Precision: ",precision2
print "Recall: ",recall2

print "AdaBoostClassifier finalizado"
print "Resultados do AdaBoostClassifier: "
print "Acuracia: ", accuracy4
print "Precision: ",precision4
print "Recall: ",recall4

"""
pca = PCA(n_components=6)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)

parameters = {'min_samples_split':[2,20,50]}
svr = DecisionTreeClassifier()
clf3 = GridSearchCV(svr, parameters)
clf3 = clf3.fit(features_train,labels_train)
par = clf3.best_params_["min_samples_split"]
clf3 = DecisionTreeClassifier(min_samples_split=par,random_state=0)
clf3.fit(features_train,labels_train)
pred3 = clf3.predict(features_test)
accuracy3 = accuracy_score(labels_test, pred3,normalize = True)
precision3 = precision_score(labels_test, pred3)
recall3 = recall_score(labels_test, pred3)
print "DecisionTreeClassifier finalizado"
print "Resultados do Decision Tree: "
print "Acuracia: ", accuracy3
print "Precision: ",precision3
print "Recall: ",recall3

#clf = clf3

estimators = [('reduce_dim', PCA(n_components=6)), ('clf', clf3)]
clf = Pipeline(estimators)

dump_classifier_and_data(clf, my_dataset, features_list)
