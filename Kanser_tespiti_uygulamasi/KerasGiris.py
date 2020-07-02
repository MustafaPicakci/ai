from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import keras
from keras.layers import Input, Dense
from keras.optimizers import SGD

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd

veri=pd.read_csv("breast-cancer-wisconsin.data")

veri.replace('?',-99999,inplace=True)

veri=veri.drop(['1000025'],axis=1)

imp=Imputer(missing_values=-99999,strategy='mean',axis=0)
veri=imp.fit_transform(veri)

giris=veri[:,0:8]
cikis=veri[:,9]

model=Sequential()
model.add(Dense(256,))  # Dense fully connected layer için ? | imput_dim:öğrenmek için kullanılacak sütun sayısı ? 
model.add(Activation('relu')) # relu bir aktivasyon fonksiyonu. Verileri 0-1 arasına sıkıştırır. Buna normalize etmek denir. 
                              # Diğer aktivasyon fonksiyonları : tanH,sigmoid,softmax
model.add(Dense(256))     # sayılar sanırım hidden layerdeki node sayısı
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('softmax')) # Sonuncusu 'softmax' olmalı

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(giris,cikis,epochs=50,batch_size=32,validation_split=0.13) # eğitim bu adımda gerçekleşiyor. 'epochs' veriyi kaç kere analiz edeceği.
                                              # 'batch_size' aynı anda kaç bitlik veriyi hafızaya alıp çalışacağını belirliyor
                                              # 'validation_split' eğitim sırasında kullanılmayıp en son test için ayrılacak veri yüzdesi


#sonucunu bildiğimiz bir veri ile test ediyoruz (2: iyi huylu | 4: kötü huylu)
tahmin=np.array([6,1,1,1,2,1,3,1]).reshape(1,8)
print(model.predict_classes(tahmin))