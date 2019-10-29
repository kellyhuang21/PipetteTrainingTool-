from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, make_response, session
from werkzeug.utils import secure_filename
import os
import datetime
import analysis
from analysis import runAnalysis
import urllib

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'data')
MODEL_FOLDER = os.path.join(os.getcwd(), 'model')
ALLOWED_EXTENSIONS = set(['xlsx', 'xls'])

app = Flask(__name__)
app.secret_key = os.urandom(32)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    experiments = []
    dates = []
    filenames = []
    for folder in os.listdir(app.config['UPLOAD_FOLDER']):
        datafile=""
        if folder == ".DS_Store":
            continue
        for file in os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], folder)):
            if file.endswith(".xlsx") or file.endswith(".xls"):
                datafile = file
                break
        if datafile is not "":
            headers = {'date': folder, 'raw_excel_name': datafile}
            url = "/analysis?" + urllib.parse.urlencode(headers)
            filenames.append(datafile)
            dates.append(folder.split(".")[0])
            experiments.append(url)
    return render_template("index.html", experiments=experiments, dates=dates, filenames=filenames)
    #return redirect(url_for('upload_file'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'data' in request.files:
        raw_excel = request.files['data']
        if allowed_file(raw_excel.filename):
            raw_excel_name = secure_filename(raw_excel.filename)
            date = str(datetime.datetime.now())
            save_folder = os.path.join(app.config['UPLOAD_FOLDER'], date)
            os.makedirs(save_folder)
            raw_excel.save(os.path.join(save_folder, raw_excel_name))
            return redirect(url_for('analysis', date=date, raw_excel_name=raw_excel_name))
        else:
            flash('Invalid File')
            return redirect(request.url)
    return render_template("upload.html")

@app.route('/analysis', methods=['GET'])
def analysis():
    date = request.args.get('date')
    save_folder = os.path.join(app.config['UPLOAD_FOLDER'], date)
    name = request.args.get('raw_excel_name')

    #LinReg = sampleParse(name, save_folder)
    #lineplot = url_for('uploaded_file', filename="lineplot.png", date=date)

    LinReg, results, metadata, additional = runAnalysis(name, save_folder, app.config['MODEL_FOLDER'])
    lineplots = []
    for key in LinReg.keys():
        img_name = key.split("_")[0] + "_" + "lineplot.png"
        lineplots.append([url_for('uploaded_file', filename=img_name, date=date), key, LinReg[key]])

    display_dict = {}
    for key in results.keys():
        student = key.split(" ")[0]
        img_name = key + "_" + "lineplot.png"
        img_url = url_for('uploaded_file', filename=img_name, date=date)
        if student not in display_dict.keys():
            display_dict[student] = [[key, results[key], img_url, LinReg[key]]]
        else:
            display_dict[student].append([key, results[key], img_url, LinReg[key]])

    additional = [[exp[0].values.tolist(), url_for('uploaded_file', filename=exp[1], date=date)] for exp in additional]
    return render_template("upload.html", date=date.split(".")[0],
                           results = display_dict,
                           metadata=metadata[0], additional=additional)

@app.route("/<path:path>")
def images(path):
    resp = make_response(open(path).read())
    resp.content_type = "image/png"
    return resp

@app.route("/images")
def uploaded_file():
    date = request.args.get('date')
    filename = request.args.get('filename')
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], date), filename)
