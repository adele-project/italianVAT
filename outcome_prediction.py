import random

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from transformers import AutoTokenizer, BertTokenizer, AutoModel, AutoModelForMaskedLM


def get_classifiers(names):
    if 'random' in names or 'majority' in names:
        return names

    linear_classifier = LinearSVC()
    rf_classifier = RandomForestClassifier()
    nb_classifier = GaussianNB()
    kn_classifier = KNeighborsClassifier()
    svc_classifier = SVC(kernel='poly')

    classifiers = []
    if 'linearsvc' in names:
        classifiers.append(linear_classifier)
    if 'randomforest' in names:
        classifiers.append(rf_classifier)
    if 'gaussiannb' in names:
        classifiers.append(nb_classifier)
    if 'kneighbors' in names:
        classifiers.append(kn_classifier)
    if 'svc' in names:
        classifiers.append(svc_classifier)
    return classifiers


def get_embeddings(request, text, embedding):  # embedding in [tfidf, sbert]
    if embedding == "sbert":
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        X_text = pd.DataFrame(model.encode(text))
        X_request = pd.DataFrame(model.encode(request))
        sentence_embeddings = pd.concat([X_request, X_text], axis=1)
    elif embedding == "tfidf":
        vectorizer = TfidfVectorizer()
        X_text = pd.DataFrame(vectorizer.fit_transform(text).todense(), columns=vectorizer.get_feature_names())
        X_request = pd.DataFrame(vectorizer.fit_transform(request).todense(), columns=vectorizer.get_feature_names())
        sentence_embeddings = pd.concat([X_request, X_text], axis=1)
    else:
        print("wrong embedding name")
        return
    return sentence_embeddings


def predict_outcome(df, classifiers, embeddings, optionals=[]):
    random.seed(42)
    for embedding in embeddings:
        reqs = df['request'].values
        
        texts = df['args'].astype(str) + df['claims'].astype(str)
        if 'mot' in optionals:
            texts += df['mots'].astype(str)
            texts += df['mots_of_claims'].astype(str)
        if 'dec' in optionals:
            texts += df['decs'].astype(str)
        texts = texts.values
        sentence_embeddings = get_embeddings(reqs, texts, embedding)

        for classifier in get_classifiers(classifiers):
            y_pred_all = None
            y_test_all = None
            for fold in range(1, 6):
                X_train = sentence_embeddings[df['split'] != fold]
                y_train = df[df['split'] != fold]['outcome']

                X_test = sentence_embeddings[(df['split'] == fold) & (df['grade'] == 2)]
                y_test = df[(df['split'] == fold) & (df['grade'] == 2)]['outcome']

                if classifier == 'random':
                    labs = list(set(y_train))
                    y_pred = [random.choice(labs) for _ in range(len(X_test))]
                elif classifier == 'majority':
                    labs = set(y_train)
                    maj = 0
                    for l in labs:
                        val = list(y_train).count(l)
                        if val > maj:
                            majority_class = l
                            maj = val
                    y_pred = [majority_class for _ in range(len(X_test))]
                else:
                    classifier.fit(X_train, y_train)
                    y_pred = classifier.predict(X_test)

                y_pred_all = y_pred if y_pred_all is None else np.concatenate([y_pred_all, y_pred])
                y_test_all = y_test if y_test_all is None else np.concatenate([y_test_all, y_test])

            report = classification_report(y_test_all, y_pred_all, target_names=['0', '1'])
            print(embedding + " " + str(classifier.__class__).split('.')[-1].split("'")[0])
            print(report)
