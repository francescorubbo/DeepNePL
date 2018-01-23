from flask import Flask
import os
from flask_uploads import UploadSet, configure_uploads, ALL, patch_request_class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'The secretiest of passphrases'
app.config['UPLOADED_DATAFILES_DEST'] = os.getcwd()+'/data/uploads/'

datafiles = UploadSet('datafiles', ALL)
configure_uploads(app, datafiles)
patch_request_class(app)  # set maximum file size, default is 16MB

from deepnepl import views
