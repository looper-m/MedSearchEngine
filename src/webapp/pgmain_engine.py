from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from src.webapp import *
from collections import defaultdict
from src.searchmodel import TfIdf
import xml.etree.ElementTree as xml
import matplotlib.pyplot as plt
import numpy as np
import heapq
import os
import re

app = Flask(__name__, static_folder=UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = TfIdf()


def interpolate(precision):
    max_r = -1
    for index in reversed(range(len(precision))):
        if precision[index] <= max_r:
            precision[index] = max_r
        else:
            max_r = precision[index]


def evaluate_engine() -> float:
    # Parse CF Query document
    root = xml.parse(os.path.join(DOCUMENT_SRC_FOLDER, "cfquery.xml")).getroot()
    queries = dict()
    query_attributes = defaultdict(dict)
    for document in root:
        query_no = int(document.find("QueryNumber").text)
        for attribute in document:
            if attribute.tag == "QueryText":
                queries[query_no] = attribute.text
            elif attribute.tag == "Results":
                query_attributes[query_no][attribute.tag] = int(attribute.text)
            elif attribute.tag == "Records":
                query_attributes[query_no].setdefault(attribute.tag, list())
                for item in attribute:
                    query_attributes[query_no][attribute.tag].append(int(item.text))

    precision_10 = list()
    average_precision_10 = 0
    mean_interpolated_recall = [0] * 1239
    mean_interpolated_precision = [0] * 1239
    queries_len = len(queries)
    k = 1239
    # Iterate through all queries, compute TfIdf for every query term,
    # find the cosine similarity and rank the retrieved documents
    for query_no, query in queries.items():
        ranking_heap = list()
        for doc_id, norm in model.l2_norm.items():
            dot_prod = 0
            for word in filter(lambda token: len(token) > 0, re.split('[^a-z0-9\']', query.lower())):
                value = model.tf_idf[word]
                if doc_id in value:
                    dot_prod += value[doc_id]
            # we use a min heap but negate the score to simulate a max heap
            heapq.heappush(ranking_heap, (-dot_prod / norm, doc_id))

        relevant_docs_count = query_attributes[query_no]["Results"]
        relevant_docs = query_attributes[query_no]["Records"]
        relevant_count = 0
        precision_10_relevant_count = 0
        cumulative_precision_10 = 0

        for i in range(k):
            doc_id = heapq.heappop(ranking_heap)[1]
            if doc_id in relevant_docs:
                relevant_count += 1
                if i <= 10:
                    cumulative_precision_10 += relevant_count / (i + 1)
                    precision_10_relevant_count += 1
            mean_interpolated_precision[i] += relevant_count / (i + 1)
            mean_interpolated_recall[i] += relevant_count / relevant_docs_count

        interpolate(mean_interpolated_precision)
        precision_10.append(precision_10_relevant_count / 10)
        average_precision_10 += cumulative_precision_10 / precision_10_relevant_count

    for i in range(k):
        mean_interpolated_recall[i] /= queries_len
        mean_interpolated_precision[i] /= queries_len

    mean_average_precision = average_precision_10 / queries_len

    print("MAP", mean_average_precision)
    print("PRECISION ", mean_interpolated_precision)
    print("RECALL ", mean_interpolated_recall)

    # If invoked through web server, a new thread is spawned to handle Matplotlib plots. Matplotlib is not
    # thread safe and will cause the server to crash. For the sake of the assignment, the plot has been
    # prepared before and is simply displayed. If you wish to verify, run this function locally or simply
    # check the logs for Precision, Recall prints
    # plot_curves(precision_10, mean_interpolated_precision, mean_interpolated_recall)

    return mean_average_precision


def plot_curves(precision_10, mean_interpolated_precision, mean_interpolated_recall):
    """
    Plot Precision@10 and averaged Precision Recall curves.

    :param precision_10: precision@10 values for all queries
    :param mean_interpolated_precision: averaged P@K for all queries
    :param mean_interpolated_recall: averaged R@K for all queries
    """

    fig = plt.figure(figsize=(10, 9))
    x_precision_10 = np.array(precision_10)
    plt.subplot(2, 1, 1)
    plt.plot(x_precision_10, "b-o")
    plt.xlabel("Query")
    plt.ylabel("Precision")
    plt.title("Precision@10 for 100 queries")

    x_all_avg_recall = np.array(mean_interpolated_recall)
    y_all_avg_precision = np.array(mean_interpolated_precision)
    plt.subplot(2, 1, 2)
    plt.plot(x_all_avg_recall, y_all_avg_precision, "r-o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Averaged PR curve for 100 queries")
    fig.tight_layout()
    plt.savefig(os.path.join(UPLOAD_FOLDER, "plot.png"), dpi=200)


def evaluate_query(query):
    ranking_heap = list()
    for doc_id, norm in model.l2_norm.items():
        dot_prod = 0
        for word in filter(lambda token: len(token) > 0, re.split('[^a-z0-9\']', query.lower())):
            value = model.tf_idf[word]
            if doc_id in value:
                dot_prod += value[doc_id]
        # we use a min heap but negate the score to simulate a max heap
        heapq.heappush(ranking_heap, (-dot_prod / norm, doc_id))

    k = 20
    retrieved_documents = list()
    for i in range(k):
        retrieved_documents.append(f'{heapq.heappop(ranking_heap)[1]:05}')

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'search_result.html'), 'w') as fo:
        fo.write(render_template("search_result.template.html", query=query, documents=retrieved_documents))


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        query = request.form['search-box']
        evaluate_query(query)
        return redirect(url_for('serve_search_results'))

    return render_template("engine.template.html")


@app.route('/evaluate-engine', methods=['POST'])
def on_evaluate():
    map_value = evaluate_engine()
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'evaluation_result.html'), 'w') as fo:
        fo.write(render_template("evaluate.template.html", map_value=map_value))
    return redirect(url_for('serve_evaluation_results'))


@app.route('/view/search-results')
def serve_search_results():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'search_result.html')


@app.route('/view/evaluation-results')
def serve_evaluation_results():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'evaluation_result.html')


if __name__ == "__main__":
    model.compute(DOCUMENT_SRC_FOLDER, "input_docs.txt", "stoplist.txt")
    app.run(host="localhost", port=8080)
