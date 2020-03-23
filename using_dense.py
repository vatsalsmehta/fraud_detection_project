import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import os
from datetime import datetime 
from sklearn.metrics import roc_auc_score as auc 
import seaborn as sns


df = pd.read_csv('creditcard.csv')


#for data visualization of dataset mostly sab kam aayega:
print("\n\n\n\n\n\n\n\Data Visualization begins now for beginners to understand the dataset better\n\n\n\n\n")
print(df.shape)#tocheck the shape of dataset ie. number of rows and columns
print(df.head)#basically prints the whole dataset
print(df.columns)#shows all the names of coulmns
print(df.dtypes)#shows data type (of all entities in columns)
print(df.corr())#shows correlation between data

#using scatter to plot graph of two entities of dataset
#here i plotted graph of column Time and Amount
# create a figure and axis
fig, ax = plt.subplots()
# scatter the sepal_length against the sepal_width
ax.scatter(df['Time'], df['Amount'])
# set a title and labels
ax.set_title('creditcard fraud dataset visualization. Close this graph to proceed forward and do same for all the next vsualizations')
ax.set_xlabel('Time')
ax.set_ylabel('Amount')
plt.show()


# create figure and axis
fig, ax = plt.subplots()
# plot histogram
ax.hist(df['Time'])
# set title and labels
ax.set_title('visualization')
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
plt.show()

#for histogram
plt.figure(figsize=(12,5*4))
gs = gridspec.GridSpec(5, 1)
for i, cn in enumerate(df.columns[0:1]):#o:1 0th column ka print karega as upperbound is neglected in syntax
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('Time')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()




X = df.iloc[:, 1:30].values 
y = df.iloc[:, -1].values





print("Total train examples: {}, total fraud cases: {}, equal to {:.5f} of total cases. ".format(X.shape[0], np.sum(y), np.sum(y)/X.shape[0]))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 29))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


    


