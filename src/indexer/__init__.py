from src.tokenizer import TokenizeMode
import math


def generate_inverted_index(tokens, mode=TokenizeMode.BASIC) -> dict:
    """Generate inverted index

    :argument
        tokens: list of (term, doc_id) values
        mode: inverted index mode
    :returns
        dict: inverted index dictionary (map)
    """
    inverted_index = dict()
    if len(tokens) == 0:
        return inverted_index

    if mode == TokenizeMode.TF_IDF:
        for token in tokens:
            term, doc_id = token[0], token[1]

            # update document frequency
            inverted_index.setdefault(term, [0, dict()])
            if doc_id not in inverted_index[term][1]:
                inverted_index[term][0] += 1

            # add/update term frequency for a document
            inverted_index[term][1].setdefault(doc_id, 0)
            inverted_index[term][1][doc_id] += 1
    else:
        for token in tokens:
            term, doc_id = token[0], token[1]

            # update document frequency
            inverted_index.setdefault(term, [0, list()])
            inverted_index[term][0] += 1

            # update postings
            inverted_index[term][1].append(doc_id)

    return inverted_index
