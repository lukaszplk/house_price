import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error,mean_absolute_error

df=pd.read_csv("housing.csv")
df.dropna(inplace=True)
df=df.drop('ocean_proximity', axis=1)

X = df.drop('median_house_value',axis=1)
y = df['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

model.fit(X_train,y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=100,epochs=300, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

predictions = model.predict(X_test)
print("Mean absolute error: ", mean_absolute_error(y_test,predictions))
print("Root mean squared error: ", mean_squared_error(y_test,predictions,squared=False))


answer = input("To predict press 1, to finish press 0. ")

while int(answer):
    list = []
    print('''\n\nEnter in turn:\n
longitude: A measure of how far west a house is; a higher value is farther west

latitude: A measure of how far north a house is; a higher value is farther north

housingMedianAge: Median age of a house within a block; a lower number is a newer building

totalRooms: Total number of rooms within a block

totalBedrooms: Total number of bedrooms within a block

population: Total number of people residing within a block

households: Total number of households, a group of people residing within a home unit, for a block

medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)''')
    for _ in range(8):
        list.append(input())
    print("Predictable median house value is equal to: ", model.predict(scaler.transform([list]))[0][0],"$")
    answer = input("To predict press 1, to finish press 0. ")


