import pandas as pd
from sklearn.preprocessing import StandardScaler

def loadDataframe(dataframePath, ):
    trainDataframe = pd.read_csv(dataframePath)

    for column in trainDataframe:
        if(column == "Classe" or column == "row ID"):
            continue
        trainDataframe[column] = (trainDataframe[column] - trainDataframe[column].min())/(trainDataframe[column].max() - trainDataframe[column].min())

    compressedDataframe = trainDataframe[[column for column in trainDataframe if column != "Classe" and column != "row ID"]]

    return trainDataframe, compressedDataframe