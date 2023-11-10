from dataFrameLoader import loadDataframe
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.decomposition import PCA

def showAnalysis():
    trainDataframe, compressedDataframe = loadDataframe("train.csv")

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

def dataframeTransform(dataframe, components):
    pca = PCA(n_components=components)
    pca.fit(dataframe)

    transformed = pca.transform(dataframe)

    print(f"{dataframe.shape} -> {transformed.shape}")

    return transformed