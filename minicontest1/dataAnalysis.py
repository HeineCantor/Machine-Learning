import pcaAnalysis
from dataFrameLoader import loadDataframe
import matplotlib.pyplot as plt

NUM_COMPONENTS = 3

trainDataframe, compressedDataframe = loadDataframe("train.csv")

#pcaAnalysis.showAnalysis()

validationDataframe = trainDataframe[trainDataframe["Classe"] == 2].sample(n=3)
trainDataframe.drop(validationDataframe.index, axis=0, inplace=True)

datas = pcaAnalysis.dataframeTransform(trainDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]], NUM_COMPONENTS)#.values.tolist()
classes = trainDataframe["Classe"].values.tolist()

pc0 = [x[0] for x in datas]
pc1 = [x[1] for x in datas]
pc2 = [x[2] for x in datas]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(pc0, pc1, pc2, c=classes)
plt.show()