import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def roc(output):
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    pred = output[:,1]
    fpr,tpr,_ = roc_curve(y,pred)
    plt.plot(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0,1])
    plt.ylim([0,1])    
    plt.savefig('deepnepl/static/data/plots/roc.jpg')

def accuracy(output):
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    predicted_labels = np.argmax(output, axis=1)
    accuracy = np.mean(y  == predicted_labels)
    return accuracy
