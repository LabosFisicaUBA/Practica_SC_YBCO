'''
Created on May 23, 2026

@author: Noemi
'''
import matplotlib.pyplot as plt

def create_figure(labels, data, title, xcol_num = 0, ycol_num = 1  ):
        #Plot the Data
        fig, ax = plt.subplots()
        x_label = labels[xcol_num]
        y_label = labels[ycol_num]
        x_data = data[xcol_num]
        y_data = data[ycol_num]
        plt.title(title)
        curve = ax.plot( x_data, y_data, label=y_label)
        fig.legend(curve,[y_label])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
    