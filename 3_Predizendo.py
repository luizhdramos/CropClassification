import random
random.seed(2)
import numpy as np
from pyrsgis import raster
import tensorflow as tf



import math



def loadData(modelFile, imageFile):
    model = tf.keras.models.load_model(modelFile)
    ds, featuresCoconut_Tree = raster.read(imageFile)
    
    return (model, ds, featuresCoconut_Tree)
    
def CNNdataGenerator(mxBands, kSize):
    mxBands = mxBands / 255.0
    nBands, rows, cols = mxBands.shape
    margin = math.floor(kSize/2)
    mxBands = np.pad(mxBands, margin, mode='constant')[margin:-margin, :, :]

    features = np.empty((rows*cols, kSize, kSize, nBands), dtype='float64')

    n = 0
    for row in range(margin, rows+margin):
        for col in range(margin, cols+margin):
            feat = mxBands[:, row-margin:row+margin+1, col-margin:col+margin+1]

            b1, b2, b3, b4 = feat
            feat = np.dstack((b1, b2, b3, b4))

            features[n, :, :, :] = feat
            n += 1
            
    return(features)

def createImage(model, ds, outFile, new_features):
    with tf.device('/cpu:0'):
        newPredicted = model.predict(new_features)
        newPredicted = newPredicted[:,1]
        prediction = np.reshape(newPredicted, (ds.RasterYSize, ds.RasterXSize))
        raster.export(prediction, ds, filename=outFile, dtype=str(new_features.dtype))
        

if __name__ == '__main__':
    chipSize = 5
    epochsNumber = 20
    modelFile = r'/Users/luizramos/Repos/ProjetoRPA/Data/trained_models_5by5_20Epochs/200409_CNN_Builtup_5by5_20Epochs_PScore0.681_RScore0.211_FScore0.323.h5'
    imageFile = r'/Users/luizramos/Repos/ProjetoRPA/Data/ImagemVIG.tif'
    trainedModelDir = 'trained_models_' + str(chipSize) + 'by' + str(chipSize) + '_' + str(epochsNumber) + 'Epochs'
    outFile = 'ImagemVIG_predicted_' + str(chipSize) + 'x' + str(chipSize) + '_' + str(epochsNumber) + 'Epochs.tif'
    

    
    model, ds, featuresCoconut_Tree = loadData(modelFile, imageFile)
    new_features = CNNdataGenerator(featuresCoconut_Tree, chipSize)
    createImage(model, ds, outFile, new_features)
    
