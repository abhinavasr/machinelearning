import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_graph(train, predicts, new_predicts, name):
    X_MAIN = train['X']
    Y_MAIN = train['Y']

    X = [7,8,9,10,11]
    Y = new_predicts

    X1 = X_MAIN.values
    Y1 = predicts


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.xlabel("X")
    plt.ylabel("Y")
    ax1.scatter(X_MAIN.values, Y_MAIN.values, s=10, c='b', marker="s", label='Orginal Dataset')
    #ax1.scatter(X, Y, s=10, c='burlywood', marker="s", label='Value 7,8,9,10,11')
    ax1.scatter(X1, Y1, s=10, c='g', marker="s", label='Recalculate Y based on Original X')

    plt.legend(loc='lower right')
    plt.savefig(name)