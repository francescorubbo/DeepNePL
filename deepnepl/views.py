from flask import render_template
from flask import request
from deepnepl import app
from deepnepl.models import shallowCNN

@app.route('/')
@app.route('/index')
def index():
       user = { 'nickname': 'Miguel' } # fake user
       return render_template("index.html", title = 'Home', user = user)


@app.route('/input')
def interface_input():
       return render_template("input.html")

@app.route('/output')
def interface_output():
       the_result = shallowCNN()
       return render_template("output.html", the_result = the_result)

