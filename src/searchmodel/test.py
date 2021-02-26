# from collections import defaultdict
# from src.searchmodel import TfIdf
# import xml.etree.ElementTree as xml
# import matplotlib.pyplot as plt
# import numpy as np
# import heapq
# import os
# import re
#
#
# def interpolate(precision):
#     max_r = -1
#     for index in reversed(range(len(precision))):
#         if precision[index] <= max_r:
#             precision[index] = max_r
#         else:
#             max_r = precision[index]
#
#
# # Initialize the model and compute TfIdf values
# # for all source documents.
# model = TfIdf()
# DOCUMENT_SRC_FOLDER = "documents"
# model.compute(DOCUMENT_SRC_FOLDER, "input_docs.txt", "stoplist.txt")
#
# # Parse CF Query document
# root = xml.parse(os.path.join(DOCUMENT_SRC_FOLDER, "cfquery.xml")).getroot()
# queries = dict()
# query_attributes = defaultdict(dict)
# for document in root:
#     query_no = int(document.find("QueryNumber").text)
#     for attribute in document:
#         if attribute.tag == "QueryText":
#             queries[query_no] = attribute.text
#         elif attribute.tag == "Results":
#             query_attributes[query_no][attribute.tag] = int(attribute.text)
#         elif attribute.tag == "Records":
#             query_attributes[query_no].setdefault(attribute.tag, list())
#             for item in attribute:
#                 query_attributes[query_no][attribute.tag].append(int(item.text))
#
# precision_10 = list()
# average_precision_10 = 0
# mean_interpolated_recall = [0] * 1239
# mean_interpolated_precision = [0] * 1239
# queries_len = len(queries)
# k = 1239
# # Iterate through all queries, compute TfIdf for every query term,
# # find the cosine similarity and rank the retrieved documents
# for query_no, query in queries.items():
#     ranking_heap = list()
#     for doc_id, norm in model.l2_norm.items():
#         dot_prod = 0
#         for word in filter(lambda token: len(token) > 0, re.split('[^a-z0-9\']', query.lower())):
#             value = model.tf_idf[word]
#             if doc_id in value:
#                 dot_prod += value[doc_id]
#         # we use a min heap but negate the score to simulate a max heap
#         heapq.heappush(ranking_heap, (-dot_prod / norm, doc_id))
#
#     relevant_docs_count = query_attributes[query_no]["Results"]
#     relevant_docs = query_attributes[query_no]["Records"]
#     relevant_count = 0
#     precision_10_relevant_count = 0
#     cumulative_precision_10 = 0
#
#     for i in range(k):
#         doc_id = heapq.heappop(ranking_heap)[1]
#         if doc_id in relevant_docs:
#             relevant_count += 1
#             if i <= 10:
#                 cumulative_precision_10 += relevant_count / (i + 1)
#                 precision_10_relevant_count += 1
#         mean_interpolated_precision[i] += relevant_count / (i + 1)
#         mean_interpolated_recall[i] += relevant_count / relevant_docs_count
#
#     interpolate(mean_interpolated_precision)
#     precision_10.append(precision_10_relevant_count / 10)
#     average_precision_10 += cumulative_precision_10 / precision_10_relevant_count
#
# for i in range(k):
#     mean_interpolated_recall[i] /= queries_len
#     mean_interpolated_precision[i] /= queries_len
#
# mean_average_precision = average_precision_10 / queries_len
#
# print("MAP", mean_average_precision)
# print("PRECISION ", mean_interpolated_precision)
# print("RECALL ", mean_interpolated_recall)
#
# # Plot Precision@10 and an averaged Precision Recall curve.
# # You might witness a smooth PR curve in contrast to a jagged
# # curve. That is because the values are averaged over a 100
# # queries. A single query's curve will still look jagged
# # assuming the engine did not correctly predict all 10 truth
# # values.
# fig = plt.figure(figsize=(10, 9))
# x_precision_10 = np.array(precision_10)
# plt.subplot(2, 1, 1)
# plt.plot(x_precision_10, "b-o")
# plt.xlabel("Query")
# plt.ylabel("Precision")
# plt.title("Precision@10 for 100 queries")
#
# x_all_avg_recall = np.array(mean_interpolated_recall)
# y_all_avg_precision = np.array(mean_interpolated_precision)
# plt.subplot(2, 1, 2)
# plt.plot(x_all_avg_recall, y_all_avg_precision, "r")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Averaged 11-point Precision-Recall Graph for 100 queries")
# fig.tight_layout()
# plt.savefig("./documents/plots.png", dpi=200)
