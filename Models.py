from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras
import pandas as pd

def createNN(neuron_pctg,lr,layer_pctg,dropout):
    model = Sequential()
    nneurons = float(neuron_pctg) * 40  # número máximo de neuronas por capa
    nlayers = float(layer_pctg)*50 # número máximo de capas ocultas


    layer_counter = 0
    model.add(Dense(35,activation="sigmoid",input_shape=(34,)))  # input layer
    while layer_counter < nlayers:  # hidden layer limited to
        model.add(Dense(nneurons, activation="sigmoid"))
        model.add(Dropout(dropout))
        layer_counter += 1
    model.add(Dense(1,activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy',optimizer=opt,metrics=["accuracy"])
    return model


def evaluateModel(model,x_train,y_train,test_set,batch_size):
    y_test = test_set[['Label']]
    x_test = test_set.loc[:,test_set.columns!='Label']
    model.fit(x_train,y_train,verbose=1,epochs=10,batch_size=batch_size)
    pred = model.predict(x_test)
    pred = list(round(i[0]) for i in pred)
    out =pd.DataFrame()
    out["Date"]=test_set.index
    change_pred = lambda pred : "buy" if pred ==1 else "sell"
    out["Predictions"]=list(change_pred(i)for i in pred)

    return out





