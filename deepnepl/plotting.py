import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def roc(outputs):
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    for output in outputs:
        pred = output[:,1]
        fpr,tpr,_ = roc_curve(y,pred)
        plt.plot(fpr,tpr)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.xlim([0,1])
    plt.ylim([0,1])    
    plt.savefig('deepnepl/static/data/plots/roc.jpg')
    plt.clf()

def accuracy(outputs):
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    accuracies = []
    for output in outputs:
        predicted_labels = np.argmax(output, axis=1)
        accuracy = np.mean(y  == predicted_labels)
        accuracies.append(accuracy)
    return accuracies

def plotaccuracy(outputs):
    
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    for output in outputs:
        predicted_labels = np.argmax(output, axis=1)
        accuracy = np.mean(y  == predicted_labels)
        error = np.sqrt(accuracy*(1-accuracy)/len(y))
        plt.errorbar([0],[accuracy],[error])

    medianacc = 0.68
    iqracc = 0.2
    plt.fill_between([-1,1],[medianacc-iqracc]*2,[medianacc+iqracc]*2,alpha=0.3)
    plt.axhline(medianacc,color='r')
    plt.xlim([-1,1])
    plt.ylim([0,1])
    plt.savefig('deepnepl/static/data/plots/accuracy.jpg')
    plt.clf()
