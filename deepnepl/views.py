from flask import render_template
from flask import request
from deepnepl import app, datafiles
from deepnepl.models import shallowCNN

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField

class UploadForm(FlaskForm):
    datafile = FileField(u'Test dataset',validators=[FileRequired(u'File was empty!')])
    submit = SubmitField(u'Upload')

@app.route('/', methods=['GET', 'POST'])
def input():
       form_data = UploadForm()
       form_label = UploadForm()
       for form in [form_data,form_label]:
           if form.validate_on_submit():
               filename = datafiles.save(form.datafile.data)
       return render_template("input.html", form_data=form_data, form_label=form_label)

@app.route('/output')
def output():
       the_result = shallowCNN()
       return render_template("output.html", the_result = the_result)

@app.route('/retrain')
def retrain():
       return render_template("input.html")
