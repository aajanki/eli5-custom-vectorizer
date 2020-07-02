import gzip
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.validation import check_is_fitted
from gensim.models.keyedvectors import KeyedVectors


class SIFTransformer(TransformerMixin, BaseEstimator):
    """Smoothed Inverse Frequency weighting scheme
    Sanjeev Arora and Yingyu Liang and Tengyu Ma: A Simple but
    Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017
    """

    def __init__(self, *, word_freq_filename, word2vec_filename, a=5e-4,
                 token_pattern=r'(?u)\b\w\w+\b'):
        self.a = a
        self.unigram_probabilities = self._load_unigram_probabilities(word_freq_filename)
        self.w2v = KeyedVectors.load_word2vec_format(word2vec_filename, binary=True)
        self.token_pattern = token_pattern
        self.tokenizer = re.compile(self.token_pattern).findall

    def fit(self, documents, y=None):
        X = np.array([self.embedding(doc) for doc in documents])
        self.principal_component_ = self._compute_first_pc(X)

        return self

    def transform(self, documents):
        check_is_fitted(self, attributes=['principal_component_'],
                        msg='The principal component vector is not fitted')

        X = np.array([self.embedding(doc) for doc in documents])
        return self._remove_pc(X, self.principal_component_)

    def embedding(self, text):
        vecs = []
        for w in self.tokenizer(text):
            emb = self.word_embedding(w)
            w = self.a/(self.a + self.unigram_probabilities.get(w, 0.0))
            vecs.append(w*emb)

        if vecs:
            return np.atleast_2d(np.array(vecs)).mean(axis=0)
        else:
            return np.zeros(self.w2v.vector_size)

    def word_embedding(self, word):
        if word in self.w2v.vocab:
            return self.w2v.word_vec(word)
        elif word.lower() in self.w2v.vocab:
            return self.w2v.word_vec(word.lower())
        else:
            return np.zeros(self.w2v.vector_size)

    def _load_unigram_probabilities(self, filename):
        freqs = []
        tokens = []

        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split(' ', 1)
                freqs.append(int(arr[0]))
                tokens.append(arr[1])

        freqs = np.array(freqs)
        p = freqs/freqs.sum()
        return dict(zip(tokens, p))

    def _compute_first_pc(self, X):
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def _remove_pc(self, X, pc):
        return X - X.dot(pc.transpose()) * pc
