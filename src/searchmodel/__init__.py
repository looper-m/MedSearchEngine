from collections import defaultdict
from src import indexer
from src.tokenizer import Tokenizer, TokenizeMode
import re
import os
import math


class TfIdf:
    """Represents TF_IDF model and its methods"""

    def __init__(self):
        self.tf_idf = defaultdict(dict)
        self.l2_norm = defaultdict(int)
        self.mode = TokenizeMode.TF_IDF

    def compute(self, documents_path, documents_listfile, stopwords_document=None):
        """
        Invokes Tokenizer and Indexer in TF_IDF mode, fetches
        the enhanced inverted index and computes TF_IDF values
        for every term, document.

        :param documents_path: path to documents
        :param documents_listfile: document containing the paths of
               all documents to be parsed
        :param stopwords_document: document containing all the
               stop-words
        :return:
        """
        tokenizer = Tokenizer(self.mode)

        # initialize stop list words
        stoplist_path = os.path.join(documents_path, stopwords_document)
        if stopwords_document is not None and os.path.isfile(stoplist_path):
            tokenizer.init_stoplist(stoplist_path)

        # tokenize source documents in tf-idf mode
        tokens = list()
        with open(os.path.join(documents_path, documents_listfile)) as index_document:
            content = index_document.read()
        for xml_document in filter(None, re.split('[,;:/\-\s\r\n\t]+', content)):
            tokens.extend(tokenizer.tokenize_xml(os.path.join(documents_path, xml_document)))

        # generate enhanced inverted index with term and document frequencies
        en_inverted_index = indexer.generate_inverted_index(tokens, self.mode)

        # normalize term frequency and compute inverse document frequency; update the index
        for term, value in en_inverted_index.items():
            value[0] = math.log10(len(tokenizer.documents_sz) / value[0])
            for doc_id, freq in value[1].items():
                value[1][doc_id] = 1 + math.log10(freq)

        # compute tf * idf
        for term, value in en_inverted_index.items():
            idf = value[0]
            for doc_id, tf in value[1].items():
                self.tf_idf[term][doc_id] = tf * idf

        # compute L2 norm of documents
        self.compute_l2_norm()

    def compute_l2_norm(self):
        """Computes l2 norm of all documents"""
        for term, value in self.tf_idf.items():
            for doc_id, tf_idf_value in value.items():
                self.l2_norm[doc_id] += tf_idf_value ** 2

        for doc_id, value in self.l2_norm.items():
            self.l2_norm[doc_id] = math.sqrt(value)
