from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
from deepnepl import app, datafiles
from deepnepl.models import shallowCNN, CSPLDA
import deepnepl.plotting as plotting

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/retraining')
def retraining():
    return render_template('index.html')

@app.route('/testing')
def testing():
    return render_template('testing.html')

@app.route('/uploader/<mode>/<filename>', methods=['GET', 'POST'])
def upload_file(mode,filename):
    if request.method == 'POST':
        print( request.form )
        f = request.files['file']
        f.save('deepnepl/static/data/uploads/'+secure_filename(filename))
        print('file uploaded successfully')
    return render_template(mode+".html")
                                   
@app.route('/output')
def output():
       output = shallowCNN()
       print(CSPLDA())
       the_result = plotting.accuracy(output)
       plotting.roc(output)
       return render_template("output_testing.html", the_result = the_result)

@app.route('/retrain')
def retrain():
       return render_template("input.html")
