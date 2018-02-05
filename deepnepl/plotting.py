from cycler import cycler
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import time
import os
import glob
from matplotlib.collections import LineCollection

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

    medianaccs = {'Model A': 72, 'Model B': 77.9, 'Model C': 79.4}
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-','--','-.','.']
    for label,col,ls in zip(labels,colors,linestyles):
        plt.axhline(medianaccs[label],color=col,linestyle=ls,label=label+' median')
    plt.axhline(50,color='gray',linestyle='--',label='Chance')
    #    plt.fill_between([x[0]-0.5,x[-1]+0.5],[lowacc]*2,[highacc]*2,alpha=0.3,label='Model A IQR')
    plt.xticks(x, labels)
    plt.ylabel('Accuracy [%]')
    plt.xlim([x[0]-0.5,x[-1]+0.5])
    plt.ylim([0,100])
    plt.grid(True)
    plt.legend(loc='lower left')
    for f in glob.glob('deepnepl/static/data/plots/*'):
        os.remove(f)
    filename = 'accuracy.jpg'+str(time.time())
    plt.savefig('deepnepl/static/data/plots/'+filename,format='jpg')
    plt.clf()
    plt.close()
    return filename


def plotdiagnostics(numRows=4):
    X = np.load('deepnepl/static/data/uploads/test_X.npy')
    X = X[0:2]

    filenames = []
    for ii,data in enumerate(X):
        print('looping',ii)
        fig, axarr = plt.subplots(4,4,sharex=True,figsize=(30,18))
        dmin = data.min()
        dmax = data.max()
        dr = (dmax - dmin) * 0.7  # Crowd them a bit.
        y0 = dmin
        y1 = (numRows - 1) * dr + dmax *0.7
        times = np.linspace(0.,3.1,497)

        for xax,axs in enumerate(axarr):
            for yax,ax in enumerate(axs):
                ax.set_ylim(y0, y1)
                segs = [np.hstack( (times[:, np.newaxis], data[i, :, np.newaxis]) )
                        for i in range(4*numRows*xax+numRows*yax,4*numRows*xax+numRows*yax+numRows)]
                ticklocs = [i * dr for i in range(numRows)]
                offsets = np.zeros((numRows, 2), dtype=float)
                offsets[:, 1] = np.array(ticklocs)
                lines = LineCollection(segs, offsets=offsets, transOffset=None)
                for off in offsets:
                    ax.axhline(off[1],color='gray',linestyle='--')
                ax.add_collection(lines)
                
                ax.set_yticks(ticklocs)
                ax.set_yticklabels(
                    range(4*numRows*xax+numRows*yax,4*numRows*xax+numRows*yax+numRows))
                
                ax.set_xlim([0,3.1])
                
        filename = 'diagnostics.jpg'+str(time.time())
        plt.savefig('deepnepl/static/data/plots/'+filename,format='jpg')
        plt.clf()
        plt.close(fig)
        filenames.append(filename)
    
    return filenames
