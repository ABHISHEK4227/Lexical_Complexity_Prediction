from keras.layers import Input
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.regularizers import l2
from keras.layers import Flatten
from keras.layers import Dense, Input , Dropout
from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from tensorflow.keras.models import Model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
from scipy.stats import pearsonr,spearmanr
import numpy as np
import statistics

MAE = tf.keras.losses.MeanAbsoluteError()
MSE= tf.keras.losses.MeanSquaredError()


max_pearson_score = -1

class CustomCallback(tf.keras.callbacks.Callback):

  def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


  def on_epoch_end(self, epoch, logs=None):
    y_true=self.y_val
    y_pred_val= self.model.predict(self.x_val)
    r2_Score=r2_score(y_true,y_pred_val)    
    pearson_score=pearsonr(y_true.to_numpy().reshape(-1,),y_pred_val.reshape(-1,))

    global max_pearson_score

    print("r2_score",r2_Score)
    print("pearson_score",pearson_score[0])
    if (pearson_score[0] > max_pearson_score):
      print("Saving Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      max_pearson_score=pearson_score[0]
      self.model.save('KNNModel.h5')  

featureCount = 1389

def kerasnn_model():                                           
    # create model
    inp=Input(shape=(featureCount),name='Input')
    drop3=Dropout(0.2)(inp)
    d4=Dense(64,activation='sigmoid',name='Dense4')(drop3)


    drop4=Dropout(0.2)(d4)
    d5=Dense(32,activation='sigmoid',name='Dense5')(drop4)

    drop5=Dropout(0.2)(d5)
    d6=Dense(8,activation='sigmoid',name='Dense6')(drop5)

    output=Dense(1,activation='sigmoid',name='output')(d6)

    model=Model(inputs=[inp],outputs=[output])
    # Compile model
    
    model.compile(optimizer='adam',loss=MSE,metrics=[MAE,MSE])
    return model
    
    
class LCP(object):
    
    
    def __init__(self,fCount,type='single'):
        global featureCount
        if type == 'single':
            self.type = 1
        else:
            self.type = 2
            
        # self.featureCount = featureCount
        featureCount = fCount
            
        # A Neural Netrowrk model using Keras
        self.modelKNN = kerasnn_model()
        
        # Adaboost model using default basic estimator 
        self.modelAB = AdaBoostRegressor(random_state=19, n_estimators=80)
        
        # Adaboost using keras nn  as estimator
        self.kada = KerasRegressor(build_fn= kerasnn_model, epochs=200, batch_size=64, verbose=0)
        self.modelKS = AdaBoostRegressor(base_estimator= self.kada,random_state=19, n_estimators=16)
        
        # Bagging Regressor 
        self.modeBR = BaggingRegressor(random_state=19, n_estimators=32)
        
        # For Bagging Regressor wihh keras nn as base estimator
        self.modeBG = BaggingRegressor(base_estimator= self.kada,random_state=19, n_estimators=32)
        
        # For GradientBoostingRegressor
        self.modeKGBR = GradientBoostingRegressor(random_state=19, n_estimators=32)
        
        # for stacking 
        self.meta_model_1 = LinearRegression()
        self.meta_model_2 = LinearRegression()
        self.meta_model_3 = LinearRegression()
        self.meta_model_4 = LinearRegression()
        self.meta_model_5 = LinearRegression()
        self.meta_model_6 = LinearRegression()
        
        return
    
    def common(self,x_train,y_train,x_val,y_val):
        
        #Train Keras NN Model
        self.modelKNN.fit(x_train, y_train,validation_data=(x_val, y_val),batch_size=64,epochs=200,callbacks=[self.call_back])
        self.modelKNN = tf.keras.models.load_model('KNNModel.h5',compile=False)
        print('KNN Done!')
        # Train on Adaboost with default basic estimator
        self.modelAB.fit(x_train, y_train)
        print('AB Done!')
        self.modelKS.fit(x_train, y_train)
        print('KS Done!')
        self.modeBR.fit(x_train, y_train)
        print('BR Done!')
        self.modeBG.fit(x_train, y_train)
        print('BG Done!')
        self.modeKGBR.fit(x_train, y_train)
        print('KGBR Done!')
        return
        
    def trainall(self,x_train,y_train,x_val,y_val):
    
        # stacking
        # 1. Keras AdaBoost
        # 2. BaggingRegressor
        # 3. GradientBoosting Regressor
        # 4. AdaBoost
        # 5. KNN
        # 6. BaggingRegressor with base estimator
        self.call_back=CustomCallback((x_train,y_train),(x_val,y_val))
        
        self.common(x_train,y_train,x_val,y_val)
        valp1 = self.modelKS.predict(x_val)
        valp2 = self.modeBG.predict(x_val)
        valp3 = self.modeKGBR.predict(x_val)
        valp4 = self.modelAB.predict(x_val)
        valp5 = self.modelKNN.predict(x_val)
        valp6 = self.modeBR.predict(x_val)
        
        #Combination KS + AB + KNN
        stacked_val_pred1 = np.column_stack((valp1,valp4,valp5))        
        self.meta_model_1.fit(stacked_val_pred1,y_val)
        
        # combo KS + BG + KGBR
        stacked_val_pred2 = np.column_stack((valp1,valp2,valp3))        
        self.meta_model_2.fit(stacked_val_pred2,y_val)
        
        # combo KS + BG + KGBR + AB
        stacked_val_pred3 = np.column_stack((valp1,valp2,valp3,valp4))        
        self.meta_model_3.fit(stacked_val_pred3,y_val)
        
        # combo KS + BG + KGBR + KNN
        stacked_val_pred4 = np.column_stack((valp1,valp2,valp3,valp5))        
        self.meta_model_4.fit(stacked_val_pred4,y_val)
        
        # combo KS + BG + KGBR + AB + KNN
        stacked_val_pred5 = np.column_stack((valp1,valp2,valp3,valp4,valp5))        
        self.meta_model_5.fit(stacked_val_pred5,y_val)
        
        # Combo all
        stacked_val_pred6 = np.column_stack((valp1,valp2,valp3,valp4,valp5,valp6))        
        self.meta_model_6.fit(stacked_val_pred6,y_val)
        
        print('Training Completed !!')
        return
        

    def predict(self,x_test,model_n0 = 6):
        yp1 = self.modelKS.predict(x_test)
        yp2 = self.modeBG.predict(x_test)
        yp3 = self.modeKGBR.predict(x_test)
        yp4 = self.modelAB.predict(x_test) 
        yp5 = self.modelKNN.predict(x_test) 
        yp6 = self.modeBR.predict(x_test)
        
        if model_n0 == 1:
            stacked_test_pred = np.column_stack((yp1,yp4,yp5))
            y_meta_pred = self.meta_model_1.predict(stacked_test_pred)
        elif model_n0 == 2:
            stacked_test_pred = np.column_stack((yp1,yp2,yp3))
            y_meta_pred = self.meta_model_2.predict(stacked_test_pred)
        elif model_n0 == 3:
            stacked_test_pred = np.column_stack((yp1,yp2,yp3,yp4))
            y_meta_pred = self.meta_model_3.predict(stacked_test_pred)
        elif model_n0 == 4:
            stacked_test_pred = np.column_stack((yp1,yp2,yp3,yp5))
            y_meta_pred = self.meta_model_4.predict(stacked_test_pred)
        elif model_n0 == 5:
            stacked_test_pred = np.column_stack((yp1,yp2,yp3,yp4,yp5))
            y_meta_pred = self.meta_model_5.predict(stacked_test_pred)
        elif model_n0 == 6:
            stacked_test_pred = np.column_stack((yp1,yp2,yp3,yp4,yp5,yp6))
            y_meta_pred = self.meta_model_6.predict(stacked_test_pred)
        else:
            y_meta_pred = []
            for i in range(len(yp1)):
                tmp = []
                tmp.append(yp4[i])
                tmp.append(yp3[i])
                tmp.append(yp6[i])
                tmp.append(yp5[i])
                tmp=[float(x) for x in tmp]
                y_meta_pred.append(statistics.mean(tmp))
        return np.array(y_meta_pred)
        
    
        