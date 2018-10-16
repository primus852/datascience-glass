import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Import the Dataframe
dataset = pd.read_csv('data/glass.csv', delimiter=',', header=0)

columns = ['Ri', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']

# Convert NaN and other invalid values to mean
for column in columns:
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

# Split
X = dataset.iloc[:, 0:9]
y = dataset.iloc[:, 9]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Scale the Data
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Determine the n_neighbors (avoid using even voters)
nn = int(math.sqrt(len(y_test)))

if (nn % 2) == 0:
    nn = nn - 1

# (Check it)
print(str(math.sqrt(len(y_test))) + ' => ' + str(nn))

# Get the unique Target values
uni_target = len(dataset['Type'].unique())
print('Target Values: ' + str(uni_target))

# Init Classifier
classifier = KNeighborsClassifier(n_neighbors=nn, p=uni_target, metric='euclidean')
scores = cross_val_score(classifier, X, y, cv=9, scoring='accuracy')
print('CrossVal: ' + str(scores.mean()))

# Fit
classifier.fit(X_train, y_train)

# Predict the test set
y_pred = classifier.predict(X_test)

# Accuracy
print(accuracy_score(y_test, y_pred))
