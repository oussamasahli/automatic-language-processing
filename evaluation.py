# -*- coding: utf-8 -*-

import numpy as np
import codecs
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from random import randint
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from imblearn.under_sampling import NearMiss
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

###############################################################################
# CHARGEMENT DES DONNEES CHIRAC / MITERRAND
###############################################################################

#fic=open("AFDpresidentutf8/corpus.tache1.learn.utf8", 'r')
#lecture = fic.readlines()
#N_lignes = len(lecture)
#fic.close()

#nblignes = N_lignes
#datax = []
#datay = np.ones(nblignes)
#s=codecs.open("AFDpresidentutf8/corpus.tache1.learn.utf8", 'r','utf-8') 
#cpt = 0
#for i in range(nblignes):
#    txt = s.readline()
#    lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
#    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
#    if lab.count('M') >0:
#        datay[cpt] = -1   
#    cpt += 1
#    datax.append(txt)
    
    
def my_tokenizer(text):
    text=re.sub("(\\W)"," \\1 ",text)
    return re.split("\\s+",text)

###############################################################################
# CHARGEMENT DES DONNEES MOVIES
###############################################################################
import os.path


def read_file(fn):
    with codecs.open(fn,encoding="utf-8") as f:
        return f.read()

path = "AFDmovies/movies1000/"

data_movies_x = [] # init vide
data_movies_y = []
cpt = 0
for cl in os.listdir(path): # parcours des fichiers d'un répertoire
    for f in os.listdir(path+cl):
        txt = read_file(path+cl+'/'+f)
        data_movies_x.append(txt)
        data_movies_y.append(cpt)

    cpt += 1
    


###############################################################################
# DONNEES / AFFICHAGE
###############################################################################
 
def histo(datay):
    sns.distplot(datay, hist=True, kde=True, bins=20, color = 'blue',
                hist_kws={'edgecolor':'blue'})
    
def  hist(x):  
    plt.hist(x, bins = 2, color = 'yellow',
                edgecolor = 'red')
    plt.title('Distribution Des Classes')

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#histo(datay)
###############################################################################
# ACCURACY
###############################################################################
    
#X_train, X_test, y_train, y_test = train_test_split(datax, datay
#                                  ,test_size=0.2, random_state=1234)

#cv = CountVectorizer()
#X_train=cv.fit_transform(X_train)
#X_test=cv.transform(X_test)

#RL=LogisticRegression()
#RL.fit(X_train,y_train)
#y_pred = RL.predict(X_test) 
#print("ACCURACY: ",RL.score(X_test,y_test)*100," %")

#class_names=np.array(['classe -1', 'classe 1'])
#np.set_printoptions(precision=2)
#plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                      title='Confusion matrix, without normalization')

#plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
#                      title='Normalized confusion matrix')
#plt.show()

#histo(y_test)
    
###############################################################################
# RAPPEL / PRECISION / SCORE F1
###############################################################################    

#X_train, X_test, y_train, y_test = train_test_split(datax, datay
#                                 ,test_size=0.2, random_state=1234)

#cv = CountVectorizer()

#X_train=cv.fit_transform(X_train)
#X_test=cv.transform(X_test)

#RL = LogisticRegression(C=1,penalty='l2',max_iter=3000)
#RL.fit(X_train,y_train)
#p = RL.predict(X_test) 
#print(classification_report(y_test, p)) 

###############################################################################
# RE ECHANTILLIONNAGE ( ICI SOUS ECHANTILLIONAGE)
###############################################################################
#cv = CountVectorizer()
#X=cv.fit_transform(datax)

#nm1 = NearMiss(version=1)
#X_resampled_nm1, y_resampled = nm1.fit_resample(X, datay)
#print(sorted(Counter(y_resampled).items()))

###############################################################################
# EQUILIBRE PRESIDENT
###############################################################################
def equilibre(datax,datay):
    v=np.where(datay==1)[0]
    u=np.where(datay==-1)[0]
    X=datax
    Y=datay
    
    L=[]
    if(len(v)>len(u)):
        l=len(v)-len(u)
        for i in range(0,l):
            e=randint(0,len(v)-1)
            j=v[e]
            v=np.delete(v,e)
            L.append(j)
        Y=np.delete(Y,L)
        X=np.delete(X,L)
    else:
        l=len(u)-len(v)
        for i in range(0,l):
            e=randint(0,len(u)-1)
            j=u[e]
            u=np.delete(u,e)
            L.append(j)
        Y=np.delete(Y,L)
        X=np.delete(X,L)
    return X,Y

###############################################################################
# SOUS ECHANTILLONNAGE / F1 / RAPPEL / PRECISION
###############################################################################


#cv = CountVectorizer()
#X=cv.fit_transform(datax)
#X_resampled, y_resampled = equilibre(datax, datay)
#X_resampled=cv.transform(X_resampled)

     
#cv = CountVectorizer()
#X=cv.fit_transform(datax)
#nm = NearMiss(version=3)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)

#print(sorted(Counter(y_resampled).items()))

#X_train_eq, X_test_eq, y_train_eq, y_test_eq = train_test_split(X_resampled, y_resampled
#                                ,test_size=0.2, random_state=1234)

#RL = LogisticRegression(max_iter=3000)
#RL.fit(X_train_eq,y_train_eq)

#X_train_deseq, X_test_deseq, y_train_deseq, y_test_deseq = train_test_split(X, datay
#                                 ,test_size=0.2, random_state=1234)

#p = RL.predict(X_test_deseq) 
#print(classification_report(y_test_deseq, p)) 
    

###############################################################################
 #OPTIMISATION / PRETRAITEMENT / GRID SEARCH / PIPELINE /SVM
###############################################################################
#import time
#import nltk
#start_time = time.time()

#sw = set()
#sw.update(tuple(nltk.corpus.stopwords.words('french')))

#data_X,data_Y=equilibre(datax,datay)    
 
#pipeline = Pipeline([
#    ('vect', CountVectorizer()),
#    ('clf', SVC()),
#])
 
 
#parameters = {
#    'vect__max_df': (0.5, 0.6, 0.7, 0.8, 0.9, 1),
#    'vect__max_features': (None, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 700000,100000),
#    'vect__ngram_range': ((1, 1), (1, 2),(2,2)),  
#    'vect__min_df': (0.01, 0.05, 0.1, 0.15, 0.2, 0.25 ), 
#    'vect__lowercase': (True,False),
#    'vect__analyzer' : ('word','char_wb'),
#    'vect__stop_words': (None,sw),
#    'vect__tokenizer': (None,my_tokenizer), 
    
#    'clf__C': (0.1, 1, 10, 100, 1000),
#    'clf__gamma': (1, 0.1, 0.01, 0.001, 0.0001),
#    'clf__max_iter': (1000, 2000, 3000),
#    'clf__kernel': ('rbf','linear','poly'),
#}

#G = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,scoring='f1',cv=5)
#X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y
#                                 ,test_size=0.2, random_state=1234)
#G.fit(X_train, y_train)
#print(G.sore(X_test,y_test))
# Affichage du temps d execution
#print("Temps d execution : %s secondes ---" % (time.time() - start_time))

 
###############################################################################
# OPTIMISATION / MEILLEUR TRANSFORMATEUR
###############################################################################
#RL = LogisticRegression(max_iter=10000)
#svm= LinearSVC(max_iter=10000)
#NB= MultinomialNB()

###############################################################################
# STEMMMING
###############################################################################
#nltk.download('wordnet')
def stem(datax):
    stema=[]
    lemma= WordNetLemmatizer()
    for sent in datax:
        text=word_tokenize(sent)
        l=[lemma.lemmatize(lemma.lemmatize(lemma.lemmatize(word,pos='a'),pos='v'),pos='n') for word in text]
        stema.append(' '.join(l))
    return stema
 
###############################################################################
# STOP WORDS
###############################################################################
sw = set()
sw.update(tuple(nltk.corpus.stopwords.words('french')))

###############################################################################
# BAGS OF WORD
###############################################################################
#stema=stem(datax)
#cv = CountVectorizer(lowercase=False,ngram_range=(1,2),max_features=35000)
#X=cv.fit_transform(datax)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)

#SVM
#scores = cross_validate(svm,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#print("SVM ACCURACY:",np.mean(scores['test_score'])*100)

#RL
#scores = cross_validate(RL,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#print("RL ACCURACY:",np.mean(scores['test_score'])*100)

#NB
#scores = cross_validate(NB,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#print("NB ACCURACY:",np.mean(scores['test_score'])*100)
    
###############################################################################
# Max_fetures
###############################################################################
#s=[]
#n=[]
#r=[]
#o=[]

#for i in range (5000,105000,5000):
#    print(i)
#    cv =  CountVectorizer(binary=True,lowercase=False,max_features=i)
#    X=cv.fit_transform(datax)
#    nm = RandomUnderSampler(random_state=42) 
#    X_resampled, y_resampled = nm.fit_resample(X, datay)

#    scores = cross_validate(svm,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#    s.append(np.mean(scores['test_score'])*100)
   
#    scores = cross_validate(RL,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#    r.append(np.mean(scores['test_score'])*100)
#    
#    scores = cross_validate(NB,X, datay, cv=5,scoring='accuracy',
 #                      return_train_score=True)
 #   n.append(np.mean(scores['test_score'])*100)
  
  #  o.append(i)
    
#plt.plot(o, s, label="SVM")
#plt.plot(o, n, label="NaivesBayes")
#plt.plot(o, r, label="Regression logiciel")

#plt.xlabel("taille du vocabulaire")
#plt.ylabel("Accuracy score") 
#plt.title("Score Accuracy en fonction de la taille du vocabulaire")
#plt.legend()

#plt.show()

###############################################################################
# GRID SEARCH MIN DF /MAX /DF
###############################################################################

#X_resampled, y_resampled = equilibre(datax, datay)

#pipeline = Pipeline([
#    ('vect', CountVectorizer(binary=True,lowercase=False)),
#    ('clf', NB),
#])
 
#parameters = {
#    'vect__max_df': (0.5, 0.6, 0.7, 0.8, 0.9, 1),
 #   'vect__min_df': (0.01, 0.05, 0.1, 0.15, 0.2, 0.25 ), 
#   }

#clf = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1,scoring='accuracy',cv=5)
#clf.fit(datax, datay)
#print("Best parameter (CV score=%0.3f):" % clf.best_score_)
#print(clf.best_params_)

###############################################################################
# PRESENCE
###############################################################################
#stema=stem(datax)
#cv = CountVectorizer(binary=True,lowercase=False,ngram_range=(1,2))
#X=cv.fit_transform(datax)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)

##SVM
#scores = cross_validate(svm,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#print("SVM ACCURACY:",np.mean(scores['test_score'])*100)

#RL
#scores = cross_validate(RL,X, datay, cv=5,scoring='accuracy',
#                       return_train_score=True)
#print("RL ACCURACY:",np.mean(scores['test_score'])*100)

#NB
#scores = cross_validate(NB,X, datay, cv=5,scoring='accuracy',
#                       return_train_score=True)
#print("NB ACCURACY:",np.mean(scores['test_score'])*100)

###############################################################################
# TFIDF
###############################################################################

#stema=stem(datax)
#cv = vectorizer = TfidfVectorizer(lowercase=False,stop_words=sw)
#X=cv.fit_transform(datax)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)

#SVM
#scores = cross_validate(svm,X, datay, cv=5,scoring='accuracy',
#                        return_train_score=True)
#print("SVM ACCURACY:",np.mean(scores['test_score'])*100)

#RL
#scores = cross_validate(RL,X, datay, cv=5,scoring='accuracy',
#                       return_train_score=True)
#print("RL ACCURACY:",np.mean(scores['test_score'])*100)

#NB
#scores = cross_validate(NB,X, datay, cv=5,scoring='accuracy',
 #                      return_train_score=True)
#print("NB ACCURACY:",np.mean(scores['test_score'])*100)

###############################################################################
# GRID SEARCH
################################################################################
#RL = LogisticRegression(max_iter=10000)
#svm= LinearSVC(max_iter=3000)
#NB= MultinomialNB()

#stema=stem(datax)
#cv =  CountVectorizer(binary=True, lowercase=False,stop_words=sw)
#X=cv.fit_transform(datax)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)
    
#parameters = [{
#               'C': [0.001,0.1, 1, 10, 100, 1000],
#            }
#             ]

#clf = GridSearchCV( svm, parameters,refit = True,cv=5,scoring='accuracy')
#clf.fit(X, datay)
#print("Best parameter (CV score=%0.3f):" % clf.best_score_)
#print(clf.best_params_)




###############################################################################
# PREDICTIONS President
###############################################################################
RL = LogisticRegression()
svm= LinearSVC()
NB= MultinomialNB()


#fic=open("AFDpresidentutf8/corpus.tache1.test.utf8", 'r')
#lecture = fic.readlines()
#N_lignes = len(lecture)
#fic.close()
#nblignes = N_lignes

#test_x_pres = []
#s=codecs.open("AFDpresidentutf8/corpus.tache1.test.utf8", 'r','utf-8') 
#for i in range(nblignes):
#    txt = s.readline()
#    txt = re.sub(r"<[0-9]*:[0-9]*>(.*)","\\1",txt)
#    test_x_pres.append(txt)

#cv = TfidfVectorizer(lowercase=False,ngram_range=(1,2),tokenizer=my_tokenizer)
#X=cv.fit_transform(datax)
#nm = RandomUnderSampler(random_state=42) 
#X_resampled, y_resampled = nm.fit_resample(X, datay)
#Test_X=cv.transform(test_x_pres)

#svm.fit(X_resampled, y_resampled)
#p=svm.predict(Test_X)
#pred=[]

#for i in range (0,len(p)):
#    if(p[i]==1):
#        pred.append('C')
#    elif(p[i]==-1):
#        pred.append('M')
#pred=np.array(pred)
        
#np.savetxt('resultat_president.txt',pred,fmt="%s")


###############################################################################
# PREDICTIONS Movies
###############################################################################
 
#CHARGER DONNEE TEST MOVIES

fic=open("moviesTest/testSentiment.txt", 'r',encoding="utf8")
lecture = fic.readlines()
test_x_movies = []

for line in lecture:
    test_x_movies.append(line)


clf=LogisticRegression(C=0.001)
cv = CountVectorizer(binary=True,lowercase=False,ngram_range=(1,2))
X=cv.fit_transform(data_movies_x)
nm = RandomUnderSampler(random_state=42) 
X_resampled, y_resampled = nm.fit_resample(X, data_movies_y)
Test_movis_X=cv.transform(test_x_movies)

clf.fit(X_resampled, y_resampled)
p=clf.predict(Test_movis_X)

pred_m=[]

for i in range (0,len(p)):
    if(p[i]==1):
        pred_m.append('C')
    elif(p[i]==0):
        pred_m.append('M')
pred_m=np.array(pred_m)


np.savetxt('resulat_sentiments.txt', pred_m,fmt="%s")











