import torch
import numpy as np
from braindecode.torch_ext.util import np_to_var, var_to_np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
import sklearn

def shallowCNN():

    cuda = torch.cuda.is_available()

    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')
    net_in = np_to_var(test_X[:,:,:,None])
    if cuda:
        net_in = net_in.cuda()

    model = torch.load('deepnepl/static/data/models/shallowCNN.pth', map_location=lambda storage, loc: storage)
    if cuda:
        model.cuda()

    outputs = model(net_in)
    return var_to_np(outputs)


def CSPLDA():

    train_X = np.load('deepnepl/static/data/models/trainCSP_X.npy')
    train_y = np.load('deepnepl/static/data/models/trainCSP_y.npy')
    test_X = np.load('deepnepl/static/data/uploads/test_X.npy')
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=3, reg=None, log=True, norm_trace=False)
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    print (sklearn.__version__)
    print(train_X.shape, train_X.mean())
#    clf.fit(train_X,train_y)

    return 'ok'
#    return clf.predict_proba(test_X)
    
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



