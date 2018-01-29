from flask import render_template
from flask import request
from werkzeug.utils import secure_filename
from deepnepl import app
from deepnepl.models import CSPLDA,shallowCNN,deepCNN,deepEEGNet
import deepnepl.plotting as plotting
import numpy as np

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
    #outputCSP = CSPLDA()
    outputs = [shallowCNN(),deepCNN(),deepEEGNet()]
    labels = ['Model A','Model B','Model C']
    accuracies = plotting.accuracy(outputs)
    acc_dict = dict(zip(labels,accuracies))
    #    plotting.roc(outputs)
    plotting.plotaccuracy(outputs,labels)

    bestmodel = labels[np.argmax(accuracies)]
    recommendModel = '%s is the best performing for this patient.'%bestmodel
    warnQuality = 'All models underperform for this patience. Verify quality of EEG recordings.'
    recommendation = recommendModel if acc_dict[bestmodel]>65 else warnQuality

    return render_template("output_testing.html",
                           modelA = acc_dict['Model A'],
                           modelB = acc_dict['Model B'],
                           modelC = acc_dict['Model C'],
                           recommendation = recommendation
    )

@app.route('/retrain')
def retrain():
       return render_template("input.html")
