from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Tuple, Any
from numpy import ndarray
from scipy.sparse import spmatrix

def get_BoW(x_train : Any, x_test : Any) -> Tuple[ndarray[Any, Any] | spmatrix, ndarray[Any, Any] | spmatrix, CountVectorizer]:
    """
    Extracts Bag of Words features.
    """
    vectorizer = CountVectorizer(stop_words='english')
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec, vectorizer

def get_TF_IDF(x_train: Any, x_test: Any, ngram_range: Tuple[float, float] = (1, 1)) -> Tuple[ndarray[Any, Any] | spmatrix, ndarray[Any, Any] | spmatrix, TfidfVectorizer]:
    """
    Extracts TF-IDF features (supports n-grams).
    """
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    return x_train_vec, x_test_vec, vectorizer
