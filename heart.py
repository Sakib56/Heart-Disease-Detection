import numpy as np
from sklearn import svm

# load data
X = []
Y = []
with open("heartdataset.csv") as dataCSV:
    for i, row in enumerate(dataCSV):
        if (i > 0):
            arrayRow = map(lambda x: x.strip(), row.split(","))
            cleanedRow = list(map(lambda y: float(y), arrayRow))

            X.append(np.array(cleanedRow[:-1]))
            Y.append(np.array(cleanedRow[-1]))
            
print(X[2], Y[2])

# init svm
# classifier = svm.SVC(gamma=0.001, C=100)
# classifier.fit(X[:-1], Y[:-1])

# print("Prediction of last: ", classifier.predict(X[-1].reshape(-1,1)))
