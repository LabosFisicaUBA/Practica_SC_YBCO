'''
Created on May 23, 2026

@author: Leandro
'''

import os
from processing import data_proc


def setup( _dir = "../datos"):
    
    ## Se Posiciona en el Directorio de los datos 
    
    # dir = os.listdir("../datos/")
    # os.chdir("../datos/")
    os.chdir(_dir)
    # dir = os.listdir(".")
    # for entry in dir :
    #     print(entry)

if __name__ == '__main__':
        
    setup()
    
    ## Define el nombre de archivo a procesar
    archivo = "susceptibilidad_alterna_Hdc_0Oe_Hac_1Oe_f_1kHz.txt"
    
    labels, data = data_proc(archivo)
    for idx, label in enumerate(labels):
        print(label)
        for d_idx, val in enumerate(data[idx]):
            print("{0}\t{1}".format( d_idx, val) ) 
        print("\n")
    pass