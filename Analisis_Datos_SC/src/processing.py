'''
Created on May 23, 2026

@author: Leandro
'''

import numpy as np
from numpy import double

def data_proc(archivo):
    ## Define Estructuras de Datos
    labels = list()
    data = list()
    
    ## Abre archivo de datos
    with open(archivo, 'r' ) as fp :
        for index, line in enumerate(fp):
            ## La primera linea son las etiquetas de las columnas
            if (index == 0) :
                ## lee la informacion de las etiquetas
                labels = line.split('\t')

                for idx,_ in enumerate(labels):
                    data.append(list())
                    
            ## Las otras lineas tienen los datos a analizar
            else:
                ## Se indexa como index-1 porque la primera linea son las etiquetas
                d = line.split('\t')
                for idx, val in enumerate(d):
                    val = double(val)
                    data[idx].append(val)
                    
        ## Se transforma la columna de datos en ndarray
        for idx, col in enumerate(data):
            data[idx] = np.array(col)
        return labels, data