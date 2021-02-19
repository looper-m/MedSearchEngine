import xml.etree.ElementTree as xml
import re

stopwords = set()


def init_stoplist(stopwords_document):
    """Initialize the set of stop-words

    :argument
        stopwords_document: document containing all the stop-words
    """
    with open(stopwords_document) as document:
        words = document.read().splitlines()
        stopwords.update(words)


def clear_stoplist():
    """Clear the set of stop-words"""
    stopwords.clear()


# def tokenize_all(document) -> list:
#     """Tokenize all xml documents named in the input document
#
#     :argument
#         document: document containing the full paths of xml documents
#     :returns
#         list: tokens
#     """
#     tokens = list()
#     with open(document) as index_document:
#         content = index_document.read()
#     for xml_document in filter(None, re.split('[,;:/\-\s\r\n\t]+', content)):
#         tokens.extend(tokenize(xml_document))
#     return process_tokens(tokens)

def tokenize(xml_document) -> list:
    """Tokenize the input xml document

    :argument
        xml_document: document to parse
    :returns
        list: tokens
    """
    tokens = list()
    synthetic_document_num = 9999
    root = xml.parse(xml_document).getroot()

    for document in root:
        document_num = document.find('RECORDNUM')
        if document_num is not None:
            doc_id = int(document_num.text)
        else:
            doc_id = synthetic_document_num
        get_tokens(document, doc_id, tokens)
        synthetic_document_num += 1

    return process_tokens(tokens)


def process_tokens(tokens) -> list:
    """Removes duplicates and sorts based on term and then by doc ID

    :argument
        tokens: tokens from parsing the document
    :returns
        list: of unique(term, doc_id) and sorted tokens
    """
    processed = list(set([token for token in tokens]))
    processed.sort(key=lambda x: (x[0], x[1]))
    return processed


def get_tokens(document, doc_id, tokens):  # todo: removed references and citations for part 1
    """Recursively parses the document to obtain tokens

    :argument
        document: document to parse
        doc_id: document ID to map the token to
        tokens: returned list of tokens
    """
    for attribute in document:
        if attribute:
            get_tokens(attribute, doc_id, tokens)
        if attribute.tag != "RECORDNUM" and attribute.text is not None:
            # filter out words with apostrophe
            # i.e. doesn't
            for word in filter(lambda token: len(token) > 1 and token not in stopwords,
                               re.split('[^a-z0-9\']', attribute.text.lower())):
                # filter again by splitting at apostrophe
                # i.e. y'all'dn't've (y'all wouldn't have in Southern US dialect)
                for word_split_apostrophe in filter(lambda token: len(token) > 1 and token not in stopwords,
                                                    re.split('\'', word)):
                    tokens.append((word_split_apostrophe, doc_id))
