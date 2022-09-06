import os, random
random.seed(2)
import numpy as np
from pyrsgis import raster
from pyrsgis.ml import imageChipsFromFile


def chipSizeDefine(chipSize, output_directory):

    #Definindo chip de imagem
    chipSize = chipSize
    featuresName = 'CNN_' + str(chipSize) + 'by' + str(chipSize) + '_features.npy'
    labelsName = 'CNN_' + str(chipSize) + 'by' + str(chipSize) + '_labels.npy'
    
    return (featuresName, labelsName)

def chipsCreation (feature_file, label_file, chipSize):
    
    #Definindo arquivos
    feature_file = feature_file
    label_file = label_file
    
    #Criando chips de Feature usando Pyrgis
    features = imageChipsFromFile(feature_file, x_size=chipSize, y_size=chipSize)
    print(feature_file)
    features = np.rollaxis(features, 3, 1)
    
    #Remodelando label
    ds, labels = raster.read(label_file)
    labels = labels.flatten()
    
    #Exibindo Detalhes da operação
    print('Input features shape:', features.shape)
    print('Input labels shape:', labels.shape)
    print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
    print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))
    
    #Salvando Chips em Numpy Arrays
    np.save(featuresName, features)
    np.save(labelsName, labels)
    print('\n\n------------------------------------------') 
    print('------(Aqruivos salvo com sucesso!)-------')
    print('------------------------------------------')

if __name__ == '__main__':
    
    chipSize = 3
    feature_file = r'ImagemVIG.tif'
    label_file = r'ImagemRotulada.tif'
    
    #mudnando diretório
    output_directory = r'/Users/luizramos/Repos/ProjetoRPA/Data'
    os.chdir(output_directory)
    
   
    featuresName, labelsName = chipSizeDefine(chipSize, output_directory)
    chipsCreation (feature_file, label_file, chipSize)
    
    
    
    
    
    
    

