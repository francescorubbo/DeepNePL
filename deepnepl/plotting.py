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
        accuracies.append( int(accuracy*100) )
    return accuracies

def plotaccuracy(outputs,labels):
    y = np.load('deepnepl/static/data/uploads/test_y.npy')
    accuracies = []
    errors = []
    for output in outputs:
        predicted_labels = np.argmax(output, axis=1)
        acc = np.mean(y  == predicted_labels)
        accuracies.append( acc*100 )
        errors.append( 100*np.sqrt(acc*(1-acc)/len(y)) )
    x = list(range(len(accuracies)))
    plt.errorbar(x,accuracies,errors,fmt='o')

    medianacc = 0.78*100
    lowacc = 0.64*100
    highacc = 0.86*100
    plt.axhline(50,color='gray',linestyle='--',label='Chance')
    plt.fill_between([x[0]-0.5,x[-1]+0.5],[lowacc]*2,[highacc]*2,alpha=0.3,label='Model A IQR')
    plt.axhline(medianacc,color='r',label='Model A median')
    plt.xticks(x, labels)
    plt.ylabel('Accuracy [%]')
    plt.xlim([x[0]-0.5,x[-1]+0.5])
    plt.ylim([0,100])
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.savefig('deepnepl/static/data/plots/accuracy.jpg')
    plt.clf()
