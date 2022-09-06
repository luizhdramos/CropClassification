import os, random
random.seed(2)
import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




def load_NumpyChips(featuresName, labelsName):
    features = np.load(featuresName)
    labels = np.load(labelsName)
    return (features, labels)

def DataTransformation(features, labels):
    
    #Separando dados
    built_features = features[labels==1]
    built_labels = labels[labels==1]
    
    unbuilt_features = features[labels==0]
    unbuilt_labels = labels[labels==0]
    
    print('Number of records in each class:')
    print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))
    
    
        #Realizando Downsample da classe majoritária
    unbuilt_features = resample(unbuilt_features,
                                replace = False, # sample without replacement
                                n_samples = built_features.shape[0], # match minority n
                                random_state = 2)
    
    unbuilt_labels = resample(unbuilt_labels,
                              replace = False, # sample without replacement
                              n_samples = built_features.shape[0], # match minority n
                              random_state = 2)
    
    print('Number of records in balanced classes:')
    print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))
    
    #combinando dados separados    
    features = np.concatenate((built_features, unbuilt_features), axis=0)
    labels = np.concatenate((built_labels, unbuilt_labels), axis=0)
     
    #normalizando
    features = features / 255.0
    print('New values in input features, min: %d & max: %d' % (features.min(), features.max()))
    

def train_test_split(features, labels, trainProp):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    train_y = labels[randIndex[:sliceIndex]]
    test_y = labels[randIndex[sliceIndex:]]
    return(train_x, train_y, test_x, test_y)

def transposeFeature(train_x, test_x):
    train_x = tf.transpose(train_x, [0, 2, 3, 1])
    test_x = tf.transpose(test_x, [0, 2, 3, 1])
    print('Reshaped features:', train_x.shape, test_x.shape)
    _, rowSize, colSize, nBands = train_x.shape


    return (train_x, test_x, _, rowSize, colSize, nBands)
        
def CreateModel(chipSize, nBands):
    model = keras.Sequential()
    model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(chipSize, chipSize, nBands)))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    
    return model

def ModelRun(model, train_x, train_y, epochsNumber, batchsize):
    with tf.device('/cpu:0'):
        model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop',metrics=['accuracy'])
        history = model.fit(train_x, train_y, epochs=epochsNumber, validation_split = 0.1, use_multiprocessing = False, batch_size=batchsize)


    
    #Plotando Acuaracia Treino / Validação
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    #Plotando Loss Treino / Validação
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    return model

def predictModel(model, test_x, test_y):
   
    #predizendo
    with tf.device('/gpu:0'):
        yTestPredicted = model.predict(test_x)
        yTestPredicted = yTestPredicted[:,1]
        
    yTestPredicted = (yTestPredicted>0.6).astype(int)
    cMatrix = confusion_matrix(test_y, yTestPredicted)
    pScore = precision_score(test_y, yTestPredicted)
    rScore = recall_score(test_y, yTestPredicted)
    fScore = f1_score(test_y, yTestPredicted)
    
    print("Confusion matrix:\n", cMatrix)
    
    print("\nP-Score: %.3f, R-Score: %.3f, F-Score: %.3f" % (pScore, rScore, fScore))
    
    #Plotando Matrix Confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cMatrix, display_labels=['Não-coqueiro', 'coqueiro'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
    
    cMatrixNormalized = cMatrix.astype('float') / cMatrix.sum(axis=1)[:, np.newaxis]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cMatrixNormalized, display_labels=['Não-coqueiro', 'coqueiro'])
    disp2.plot(cmap=plt.cm.Blues)
    plt.show()
    
    return (pScore, rScore, fScore)
    
def ModelSave(trainedModelDir, model, modelFile, pScore, rScore, fScore):
    
    if not os.path.exists(os.path.join(os.getcwd(), trainedModelDir)):
        os.mkdir(os.path.join(os.getcwd(), trainedModelDir))
    
    model.save(modelFile % (pScore, rScore, fScore)) 
    

if __name__ == '__main__':
    
    #mudnando diretório
    output_directory = r'/Users/luizramos/Repos/ProjetoRPA/Data'
    os.chdir(output_directory)
    
    
    chipSize = 3
    epochsNumber = 1
    batchsize = 512
    featuresName = r'CNN_3by3_features.npy'
    labelsName = r'CNN_3by3_labels.npy'
    trainedModelDir = 'trained_models_' + str(chipSize) + 'by' + str(chipSize) + '_' + str(epochsNumber) + 'Epochs'
    modelFile = trainedModelDir + '/200409_CNN_Builtup_' + str(chipSize)+ 'by' + str(chipSize) +'_' + str(epochsNumber) + 'Epochs_PScore%.3f_RScore%.3f_FScore%.3f.h5'

    
    features, labels = load_NumpyChips(featuresName, labelsName)
    DataTransformation(features, labels)
    train_x, train_y, test_x, test_y = train_test_split(features, labels, trainProp=0.7)
    train_x, test_x, _, rowSize, colSize, nBands = transposeFeature(train_x, test_x)
    model = CreateModel(chipSize, nBands)
    model = ModelRun(model, train_x, train_y, epochsNumber, batchsize)
    pScore, rScore, fScore = predictModel(model, test_x, test_y)
    ModelSave(trainedModelDir, model, modelFile, pScore, rScore, fScore)
    
