import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras.callbacks import EarlyStopping

df=pd.read_csv(r"C:\Users\divya\Downloads\archive\healthcare-dataset-stroke-data.csv")
df.shape
df.isnull().sum()
df.head(10)
df.tail(10)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
colamnes= ['gender',	'ever_married',	'work_type',	'Residence_type',	'smoking_status']
df[colamnes] = df[colamnes].apply(LabelEncoder().fit_transform)

fig, ax = plt.subplots(figsize=(7, 5))
plt.title('Heart Disease Distribution')
sns.countplot(x=df['heart_disease'])
ax.set_xticklabels(['No heart disease', 'Heart disease'])

sns.kdeplot(data=df, x='age', hue='stroke')
plt.title('Stroke vs Age')
plt.legend(['Stroke', 'No stroke'])

from sklearn.model_selection import train_test_split

x = df.drop("stroke", axis = 1).values
y = df['stroke'].values

x_train,x_test ,y_train,y_test = train_test_split(x,y , test_size= 0.3 , random_state=42)

x_train.shape

y_train.shape

x_test.shape

y_test.shape

# Number of classes
classes=len(set(y_train))
classes

model = Sequential()
model.add(Flatten(input_shape=(x_train.shape[1],)))
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation = 'sigmoid'))
model.summary()

model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])

early_stopping_monitor = EarlyStopping(patience = 3)
model.fit(x_train,y_train,validation_split = 0.3,epochs = 100,batch_size = 100, callbacks = [early_stopping_monitor])

model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
predictions[0]
