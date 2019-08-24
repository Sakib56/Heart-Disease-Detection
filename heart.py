import numpy as np
from sklearn import svm
from random import shuffle


def loadData(fileName):
    dataTups = []
    with open(fileName) as dataCSV:
        for i, row in enumerate(dataCSV):
            if (i > 0):
                rowArray = map(lambda x: x.strip(), row.split(","))
                cleanedRow = list(map(lambda y: float(y), rowArray))
                dataTups.append((cleanedRow[:-1], cleanedRow[-1]))
    return dataTups


def crossValidate(dataTup, trainSize=0.8):
    shuffle(dataTup)

    size = int(len(dataTup)*trainSize)
    trainingDataTup = dataTup[:size]
    testingDataTup = dataTup[size:]

    trainingData = []
    trainingTargets = []
    for data, targets in trainingDataTup:
        trainingData.append(data)
        trainingTargets.append(targets)

    testingData = []
    testingTargets = []
    for data, targets in testingDataTup:
        testingData.append(data)
        testingTargets.append(targets)

    return np.array(trainingData), np.array(trainingTargets), np.array(testingData), np.array(testingTargets)


dataTup = loadData("heartdataset.csv")
trainingData, trainingTargets, testingData, testingTargets = crossValidate(dataTup)

classifier = svm.SVC(gamma=0.000043, C=700)
classifier.fit(trainingData, trainingTargets)

total = 0
correct = 0
for X, Y in zip(testingData, testingTargets):
    if classifier.predict([X]) == Y:
        correct += 1
    total += 1
print(correct/total)

#g = 4.274e-05
#c = 700
