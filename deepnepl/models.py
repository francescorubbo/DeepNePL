import torch
import numpy as np
from braindecode.torch_ext.util import np_to_var, var_to_np

def shallowCNN():

    cuda = torch.cuda.is_available()

    test_y = np.load('data/uploads/test_y.npy')
    test_X = np.load('data/uploads/test_X.npy')
    net_in = np_to_var(test_X[:,:,:,None])
    if cuda:
        net_in = net_in.cuda()

    model = torch.load('data/models/shallowCNN.pth', map_location=lambda storage, loc: storage)
    if cuda:
        model.cuda()
        
    outputs = model(net_in)
    predicted_labels = np.argmax(var_to_np(outputs), axis=1)
    print( predicted_labels )
    accuracy = np.mean(test_y  == predicted_labels)
    print( 'Accuracy is', accuracy )
    return accuracy
