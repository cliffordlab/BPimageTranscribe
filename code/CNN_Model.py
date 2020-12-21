"""
Model class: Convolutional neural network description
Functions:
    1. __init__() : Contains image arrays and ground truth labels for training(X_train, y_train) and validation data(X_val, y_val) and "model name" to save the model
    2. model_init(): Defines the model architecture
    3. train_predict(): Trains model and estimates labels on validation data
        1. train(): Train and saves the model. 
        2. plot_loss(): Saves plots on digit-by-digit and overall validation and training loss
        3. plot_acc(): Saves plots on overall validation and training accuracy
        4. predict(): Estimates labels for validation data & calculates estimation accuracy
                      Saves estimation file as validation_data_predictions.csv in 
                      format (filename |true| predicted |difference |quality) 
                      
    6.All results are saved in a user-defined results folder while calling train_predict() function                   
"""
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dropout, Activation
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras import regularizers
import keras.backend
from keras.optimizers import Adam
#matplotlib.use('agg')
import matplotlib.pyplot as plt


class Model_Multi():
    def __init__(self,X_train, y_train, X_val, y_val, model_name):
        self.X_train=X_train
        self.y_train=y_train
        self.X_val=X_val
        self.y_val=y_val
        self.y_train_vect = [self.y_train["d1"], self.y_train["d2"], self.y_train["d3"]]
        self.y_val_vect = [self.y_val["d1"], self.y_val["d2"], self.y_val["d3"]]
        self.model_name=model_name
    
    def model_init(self):

        model_input = Input((80,180,1))

        x = Conv2D(32, (3, 3), padding='same', name='conv2d_hidden_1', kernel_regularizer=regularizers.l2(0.01))(model_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_1')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(64, (3, 3), padding='same', name='conv2d_hidden_2', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_2')(x)
        x = Dropout(0.30)(x)

        x = Conv2D(128, (3, 3), padding='same', name='conv2d_hidden_3', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(3, 3),name='maxpool_2d_hidden_3')(x)
        x = Dropout(0.30)(x)

        x = Flatten()(x)

        x = Dense(256, activation ='relu', kernel_regularizer=regularizers.l2(0.01))(x)

        digit1 = (Dense(output_dim =11,activation = 'softmax', name='digit_1'))(x)
        digit2 = (Dense(output_dim =11,activation = 'softmax', name='digit_2'))(x)
        digit3 = (Dense(output_dim =11,activation = 'softmax', name='digit_3'))(x)
        
        outputs = [digit1, digit2, digit3]

        self.model = keras.models.Model(input = model_input , output = outputs)
        self.model._make_predict_function()
        
    def train(self, lr = 1e-3, epochs=100):
        optimizer = Adam(lr=lr, decay=lr/10)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer= optimizer, metrics = ['accuracy'])
        es=EarlyStopping(monitor="val_loss", mode='min', verbose=1,patience=5)
        mc=ModelCheckpoint(self.model_name, monitor='val_loss',mode='min', verbose=1, save_best_only=True)
        keras.backend.get_session().run(tf.initialize_all_variables())
        self.history = self.model.fit(self.X_train, self.y_train_vect, batch_size=50, nb_epoch=epochs, verbose=1, validation_data=(self.X_val, self.y_val_vect), callbacks=[es,mc])
        train_scores= self.model.evaluate(self.X_train, self.y_train_vect, verbose=0)
        val_scores= self.model.evaluate(self.X_val, self.y_val_vect, verbose=0)
        print('Train scores: {}'.format(train_scores))
        print('Val scores: {}' .format(val_scores))

    def plot_loss(self, results):
        for i in range(1,4):
            plt.figure(figsize=[8,6])
            plt.plot(self.history.history['digit_%i_loss' %i],'r',linewidth=0.5)
            plt.plot(self.history.history['val_digit_%i_loss' %i],'b',linewidth=0.5)
            plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Loss',fontsize=16)
            plt.title('Loss Curves Digit %i' %i,fontsize=16)
            plt.savefig(results+'Loss_Curves_Digit_%i.png' %i )
#            plt.show()

        plt.figure(figsize=[8,6])
        plt.plot(self.history.history['loss'],'r',linewidth=0.5)
        plt.plot(self.history.history['val_loss'],'b',linewidth=0.5)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves Digit',fontsize=16)
        plt.savefig(results+'Loss_Curve_Number.png')            

    def plot_acc(self,results):
        
        for i in range(1,4):
            plt.figure(figsize=[8,6])
            plt.plot(self.history.history['digit_%i_acc' %i],'r',linewidth=0.5)
            plt.plot(self.history.history['val_digit_%i_acc' %i],'b',linewidth=0.5)
            plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
            plt.xlabel('Epochs ',fontsize=16)
            plt.ylabel('Accuracy',fontsize=16)
            plt.title('Accuracy Curves Digit %i' %i,fontsize=16)
            plt.savefig(results+'Accuracy_Curves_Digit_%i.png' %i )                     

    def predict(self,results):
        print(self.X_val.shape[:2])
        self.y_pred = self.model.predict(self.X_val)
        correct_preds = 0
        
        for i in range(self.X_val.shape[0]):
            pred_list_i = [np.argmax(pred[i]) for pred in self.y_pred]
            val_list_i  = self.y_val.values[i].astype('int')
            if np.array_equal(val_list_i, pred_list_i):
                correct_preds = correct_preds + 1
        print('exact accuracy', correct_preds / self.X_val.shape[0])
            
        df=pd.DataFrame(columns=['filename','true', 'predicted', 'difference', 'quality'], index=range(self.X_val.shape[0]))    
        diff = []
        for i in range(self.X_val.shape[0]):
                pred_list_i = [np.argmax(pred[i]) for pred in self.y_pred]
                pred_number = 100* pred_list_i[0] + 10 * pred_list_i[1] + 1* pred_list_i[2]
                if pred_number<1000:
                    df.predicted[i]=pred_number
                else:
                    df.predicted[i]=pred_number-1000    
                val_list_i  = self.y_val.values[i].astype('int')
                val_number = 100*  val_list_i[0] + 10 *  val_list_i[1] + 1*  val_list_i[2]
                df.filename[i]=self.ids_val.values[i]
                df.quality[i] =self.info_val.quality.values[i]
                if val_number<1000:
                    df.true[i]=val_number
                else:    
                    df.true[i]=val_number-1000
                diff.append(val_number - pred_number)
                df.difference[i]=df.true[i]-df.predicted[i]
                print(val_number, pred_number)        
        df.to_csv(results+'validation_data_predictions.csv')
        print('difference label vs. prediction', df['difference'])
    
    def train_predict(self,results):
        self.train()
        self.plot_loss(results)
        self.plot_acc(results)
        self.predict(results)
        

