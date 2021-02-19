def generate_inverted_index(tokens) -> dict:
    """Generate inverted index

    :argument
        tokens: list of (term, doc_id) values
    :returns
        dict: inverted index dictionary (map)
    """
    inverted_index = dict()
    if len(tokens) == 0:
        return inverted_index

    for token in tokens:
        term_key = token[0]
        doc_id = token[1]
        inverted_index.setdefault(term_key, [0, []])
        inverted_index[term_key][0] += 1
        inverted_index[term_key][1].append(doc_id)

    return inverted_index
