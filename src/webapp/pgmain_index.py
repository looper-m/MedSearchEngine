from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from src.tokenizer import Tokenizer
from src import indexer
from src.webapp import UPLOAD_FOLDER
import os
import sys
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def generate_view(xml_filenames, stopword_filename):
    tokenizer = Tokenizer()
    if len(stopword_filename) != 0:
        tokenizer.init_stoplist(os.path.join(app.config['UPLOAD_FOLDER'], stopword_filename))
    else:
        tokenizer.clear_stoplist()

    start_time = time.time()
    tokens = list()
    for filename in xml_filenames:
        tokens.extend(tokenizer.tokenize_xml(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    inverted_index = indexer.generate_inverted_index(tokens)

    terms = sorted(list(inverted_index.keys()))
    execution_time = round(time.time() - start_time, 3)
    size = sys.getsizeof(inverted_index) / 1000.0
    count = len(inverted_index)

    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'inverted_index.html'), 'w') as fo:
        fo.write(render_template("inverted_index.template.html", count=count, time=execution_time, size=size, sorted_keys=terms,
                                 content=inverted_index))


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        if request.files:
            xml_file_list = request.files.getlist('xml_file')
            stopword_file = request.files['stopword_file']
            if xml_file_list[0].filename == '':
                return redirect(request.url)

            xml_filenames = list()
            for xml_file in xml_file_list:
                xml_filenames.append(secure_filename(xml_file.filename))
                xml_file.save(os.path.join(app.config['UPLOAD_FOLDER'], xml_file.filename))

            stopword_filename = ""
            if stopword_file:
                stopword_filename = secure_filename(stopword_file.filename)
                stopword_file.save(os.path.join(app.config['UPLOAD_FOLDER'], stopword_file.filename))

            generate_view(xml_filenames, stopword_filename)
            return redirect(url_for('serve_file'))

    return render_template("fileupload.template.html")


@app.route('/view/.inverted_index')
def serve_file():
    return send_from_directory(app.config['UPLOAD_FOLDER'], 'inverted_index.html')


if __name__ == "__main__":
    app.run(host="localhost", port=8080)
