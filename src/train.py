from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import Any

def train_naive_bayes(x_train: Any, y_train: Any) -> MultinomialNB:
    print(" > Training Naive Bayes...")
    model = MultinomialNB()
    model.fit(x_train, y_train)
    return model

def train_logisitc_regression(x_train: Any, y_train: Any) -> LogisticRegression:
    print(" > Training Logisitc Regrssion...")
    model = LogisticRegression()
    model.fit(x_train, y_train)
    return model

def train_svm(x_train: Any, y_train: Any) -> SVC:
    print(" > Training SVM (Linear Kernekl)...")
    model = SVC(kernel='linear')
    model.fit(x_train, y_train)
    return model
