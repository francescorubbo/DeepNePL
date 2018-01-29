import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP

def CSPLDA():
    
    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')

    X = np.load('deepnepl/static/data/models/trainCSP_X.npy')
    y = np.load('deepnepl/static/data/models/trainCSP_y.npy')
    
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    
    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    clf.fit(X,y)

    return clf.predict_proba(test_X)

import torch
import numpy as np
from braindecode.torch_ext.util import np_to_var, var_to_np

def shallowCNN():

    cuda = torch.cuda.is_available()

    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')

    model = torch.load('deepnepl/static/data/models/shallowCNN.pth',
                       map_location=lambda storage, loc: storage)
    if cuda:
        model.cuda()
    model.eval()

    if cuda:
        net_in = np_to_var(test_X[:,:,:,None])
        net_in = net_in.cuda()
        return var_to_np(model(net_in))
    else:
        return np.squeeze(np.array([var_to_np(model(np_to_var(X[None,:,:,None])))
                                    for X in test_X]))

from braindecode.models.deep4 import Deep4Net
    
def deepCNN():

    cuda = torch.cuda.is_available()

    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')
    
    n_classes = 2
    in_chans = test_X.shape[1]
    model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                     input_time_length=test_X.shape[2],
                     final_conv_length='auto').create_network()
    model.load_state_dict(torch.load('deepnepl/static/data/models/modelDeepCNN.pth',
                                     map_location=lambda storage, loc: storage))
    if cuda:
        model.cuda()
    model.eval()

    if cuda:
        net_in = np_to_var(test_X[:,:,:,None])
        net_in = net_in.cuda()
        return var_to_np(model(net_in))
    else:
        return np.squeeze(np.array([var_to_np(model(np_to_var(X[None,:,:,None])))
                                    for X in test_X]))

from braindecode.models.eegnet import EEGNet
    
def deepEEGNet():

    cuda = torch.cuda.is_available()

    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')

    n_classes = 2
    in_chans = test_X.shape[1]
    model = EEGNet(in_chans=in_chans, n_classes=n_classes,
                   input_time_length=test_X.shape[2],
                   final_conv_length='auto').create_network()
    model.load_state_dict(torch.load('deepnepl/static/data/models/modelEEGNet.pth',
                                     map_location=lambda storage, loc: storage))

    if cuda:
        model.cuda()
    model.eval()

    if cuda:
        net_in = np_to_var(test_X[:,:,:,None])
        net_in = net_in.cuda()
        return var_to_np(model(net_in))
    else:
        return np.squeeze(np.array([var_to_np(model(np_to_var(X[None,:,:,None])))
                                    for X in test_X]))

    
#def retrain(num_classes):
#
#    cuda = torch.cuda.is_available()
#
#    model = torch.load('data/models/shallowCNN.pth', map_location=lambda storage, loc: storage)
#    
#    for param in model.parameters():
#        param.requires_grad = False
#
#    num_ftrs = model.conv_classifier.in_channels
#    kernels = model.kernel_size
#    model.conv_classifier = nn.Conv2d(num_ftrs, num_classes,
#                                      kernels, bias=True))
#
#    if cuda:
#        model.cuda()
#        
#    
#    
#    test_y = np.load('data/uploads/train_y.npy')
#    test_X = np.load('data/uploads/train_X.npy')



