import numpy as np
from sklearn import svm

# load data
Xs = []
Ys = []
with open("heartdataset.csv") as dataCSV:
    for i, row in enumerate(dataCSV):
        if (i > 0):
            arrayRow = map(lambda x: x.strip(), row.split(","))
            cleanedRow = list(map(lambda y: float(y), arrayRow))

            Xs.append(cleanedRow[:-1])
            Ys.append(cleanedRow[-1])
Xs = np.array(Xs)
Ys = np.array(Ys)

# init svm
classifier = svm.SVC(gamma='auto')
i = int(4*len(Xs)/5)
classifier.fit(Xs[:i], Ys[:i])

total = 0
correct = 0 
for X,Y in zip(Xs[i:], Ys[i:]):
    if int(classifier.predict([X])) == Y:
        correct += 1
    total += 1
print(correct/total)

# print("data: {0}\ntarget: {1}\npred: {2}".format(
#     Xs[i], Ys[i], classifier.predict([Xs[i]])))