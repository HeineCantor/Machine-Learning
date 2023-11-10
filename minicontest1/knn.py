import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot

trainDataframe = pd.read_csv("train.csv")

for column in trainDataframe:
    if(column == "Classe" or column == "row ID"):
        continue
    trainDataframe[column] = (trainDataframe[column] - trainDataframe[column].min())/(trainDataframe[column].max() - trainDataframe[column].min())

compressedDataframe = trainDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]]

pca = PCA(n_components=compressedDataframe.shape[1])
pca.fit(compressedDataframe)

loadings = pd.DataFrame(pca.components_.T,
columns=['PC%s' % _ for _ in range(len(compressedDataframe.columns))],
index=compressedDataframe.columns)
print(loadings)

plot.plot(pca.explained_variance_ratio_)
plot.ylabel('Explained Variance')
plot.xlabel('Components')
plot.show()

validationDataframe = trainDataframe.sample(n=50)
trainDataframe.drop(validationDataframe.index, axis=0, inplace=True)

datas = trainDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]].values.tolist()
classes = trainDataframe["Classe"].values.tolist()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(datas, classes)

valDatas = validationDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]].values.tolist()
valClasses = validationDataframe["Classe"].values.tolist()

accuracy = 0

for element, trueLabel in zip(valDatas, valClasses):
    label = knn.predict([element])
    if(label == trueLabel):
        accuracy+=1

accuracy /= len(valDatas)

print(accuracy)

