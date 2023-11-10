from sklearn import svm
import pandas as pd
from dataFrameLoader import loadDataframe
import pcaAnalysis

NUM_COMPONENTS = 3

trainDataframe, compressedDataframe = loadDataframe("train.csv")

#pcaAnalysis.showAnalysis()

validationDataframe = trainDataframe[trainDataframe["Classe"] == 2].sample(n=250)
trainDataframe.drop(validationDataframe.index, axis=0, inplace=True)

datas = pcaAnalysis.dataframeTransform(trainDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]], NUM_COMPONENTS)#.values.tolist()
classes = trainDataframe["Classe"].values.tolist()

clf = svm.SVC(kernel='linear')
clf.fit(datas, classes)

valDatas = pcaAnalysis.dataframeTransform(validationDataframe[[column for column in validationDataframe if column != "Classe" and column != "row ID"]], NUM_COMPONENTS)#.values.tolist()
valClasses = validationDataframe["Classe"].values.tolist()

accuracy = 0

for element, trueLabel in zip(valDatas, valClasses):
    label = clf.predict([element])
    if(label == trueLabel):
        accuracy+=1

accuracy /= len(valDatas)

print(accuracy)

testDataframe, compressedTestdataframe = loadDataframe("test.csv")

testDataframe.dropna(inplace=True)

rows = testDataframe["row ID"].values.tolist()
datas = pcaAnalysis.dataframeTransform(testDataframe[[column for column in testDataframe if column != "Classe" and column != "row ID"]], NUM_COMPONENTS)

predictions = clf.predict(datas)

outputDict = {
    'ID' : rows,
    'Class' : [int(x) for x in predictions]
}

outputDataframe = pd.DataFrame(outputDict)
outputDataframe.to_csv("file.csv", index=False)