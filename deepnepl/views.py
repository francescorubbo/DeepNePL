from flask import render_template,url_for
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

@app.route('/output',methods=['GET'])
def output():
    models = {'Model A':deepEEGNet,'Model B':shallowCNN,'Model C':deepCNN}
    modelfiles={'Model A': 'modelEEGNet.pth',
                'Model B': 'shallowCNN.pth',
                'Model C': 'modelDeepCNN.pth'} 

    labels = request.args.getlist('models')
    outputs = [shallowCNN(),deepCNN(),deepEEGNet()]
    outputs = [models[label]() for label in labels]
    accuracies = plotting.accuracy(outputs)
    acc_dict = dict(zip(labels,accuracies))
    plotting.plotaccuracy(outputs,labels)

    bestmodel = labels[np.argmax(accuracies)]
    recommendModel = '%s is the best performing for this patient.'%bestmodel
    warnQuality = 'All models underperform for this patient. Verify quality of EEG recordings.'
    recommendation = recommendModel if acc_dict[bestmodel]>65 else warnQuality

    return render_template("output_testing.html",
                           acc_models = acc_dict,
                           recommendation = recommendation,
                           bestmodelpath = url_for('static',filename='data/models/'+modelfiles[bestmodel])
    )

@app.route('/retrain')
def retrain():
       return render_template("input.html")
