import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from dataFrameLoader import loadDataframe
import pcaAnalysis
import numpy as np

VALIDATION_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE

EPOCHS = 1000

TRAIN_PATH = "~/Desktop/git/Machine-Learning/minicontest1/train.csv"
TEST_PATH = "~/Desktop/git/Machine-Learning/minicontest1/test.csv"

class SoilNetworkClassifier(nn.Module):
    def __init__(self, inputSize: int, hiddenSize: int, outputClasses: int):
        super(SoilNetworkClassifier, self).__init__()

        self.fullyConnected1 = nn.Linear(inputSize, 40)
        self.dropout = nn.Dropout()
        self.fullyConnected2 = nn.Linear(40, 15)
        self.fullyConnected3 = nn.Linear(15, 15)
        self.fullyConnected4 = nn.Linear(15, 3)

    def forward(self, x: torch.Tensor):
        # x.view(x.shape[0], - 1) to remove unwanted shapes
        out = self.fullyConnected1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fullyConnected2(out)
        out = F.relu(out)
        out = self.fullyConnected3(out)
        out = F.relu(out)
        out = self.fullyConnected4(out)
        out = F.softmax(out)

        return out

class SoilDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataframe = dataFrame

    def __getitem__(self, index):
        row = self.dataframe.iloc[index][1:].to_numpy(dtype="float32")
        features = row[0:-1]
        label = torch.Tensor(np.eye(3, dtype='uint8')[int(row[-1])-1])
        rowName = self.dataframe.iloc[index][0]
        return (features, label, rowName)

    def __len__(self):
        return len(self.dataframe)

def getOneHot(value):
    label = [0, 0, 0]
    label[value-1] = 1
    return torch.Tensor(label)

def getYOutputValue(tensor):
    tensorList = tensor.tolist()[0]
    maxValue = max(tensorList[0], tensorList[1], tensorList[2])
    if(maxValue == tensorList[0]):
        return 1
    if(maxValue == tensorList[1]):
        return 2
    
    return 3

def getDatasetClassDistribution(dataFrame):
    countClass1 = dataFrame[dataFrame["Classe"] == 1]["Classe"].count()
    countClass2 = dataFrame[dataFrame["Classe"] == 2]["Classe"].count()
    countClass3 = dataFrame[dataFrame["Classe"] == 3]["Classe"].count()

    percentClass1 = countClass1/(countClass1+countClass2+countClass3) * 100
    percentClass2 = countClass2/(countClass1+countClass2+countClass3) * 100
    percentClass3 = countClass3/(countClass1+countClass2+countClass3) * 100

    print(f"CL_1: {countClass1}\t- {percentClass1} %")
    print(f"CL_2: {countClass2}\t- {percentClass2} %")
    print(f"CL_2: {countClass3}\t- {percentClass3} %")
    print("")

NUM_COMPONENTS = 4
device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

print("Dataset loading...")

trainingDataFrame, compressedDataframe = loadDataframe(TRAIN_PATH)
testDataFrame, compressedTestDataframe = loadDataframe(TEST_PATH)

trainingDataFrame = trainingDataFrame.drop([column for column in trainingDataFrame if column != "Classe" and column != "row ID"], axis=1)

pcaComponents = pcaAnalysis.dataframeTransform(compressedDataframe, NUM_COMPONENTS)

for i in range(len(pcaComponents[0])):
    trainingDataFrame.insert(i+1, f"PC{i}", [x[i] for x in pcaComponents], True)

testDataFrame["Classe"] = 0

# testDataFrame = testDataFrame.drop([column for column in testDataFrame if column != "Classe" and column != "row ID"], axis=1)
# testDataFrame.dropna(inplace=True)
# compressedTestDataframe.dropna(inplace=True)

# pcaComponents = pcaAnalysis.dataframeTransform(compressedTestDataframe, NUM_COMPONENTS)

# for i in range(len(pcaComponents[0])):
#     testDataFrame.insert(i+1, f"PC{i}", [x[i] for x in pcaComponents], True)

#randomlySelected2s = trainingDataFrame[trainingDataFrame["Classe"] == 2].sample(n = 300)
#trainingDataFrame.drop(index=randomlySelected2s.index, inplace=True)

getDatasetClassDistribution(trainingDataFrame)



trainDataset = SoilDataset(trainingDataFrame)
testDataset = SoilDataset(testDataFrame)

trainLength = int(TRAINING_PERCENTAGE * len(trainDataset))
validationLength = len(trainDataset) - trainLength

trainDataset, validationDataset = random_split(trainDataset, [trainLength, validationLength])

trainLoader = DataLoader(trainDataset, batch_size=8, shuffle=True, num_workers=2)
validationLoader = DataLoader(validationDataset, batch_size=8, shuffle=True, num_workers=2)
testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=1)

print("Creating the model...")

model = SoilNetworkClassifier(NUM_COMPONENTS, 2, 3)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.95)
#optimizer = optim.Adam(model.parameters(), lr=0.001)

model.to(device)

for epoch in range(EPOCHS):
    running_loss = 0.0
    validation_loss = 0.0

    for i, (inputs, labels, rowName) in enumerate(trainLoader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        predicted = model(inputs)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    running_loss /= i+1

    for i, (inputs, labels, rowName) in enumerate(validationLoader):
        inputs, labels = inputs.to(device), labels.to(device)
        predicted = model(inputs)
        loss = criterion(predicted, labels)
        validation_loss += loss.item()

    validation_loss /= i+1

    print(f"{epoch}\t- Train Loss: {running_loss} - Validation Loss: {validation_loss}")


outputDict = {
    'ID' : [],
    'Class' : []
}

model.eval()

for i, (inputs, labels, rowName) in enumerate(testLoader):
    inputs, labels = inputs.to(device), labels.to(device)
    yPredicted = model(inputs)
    print(yPredicted)
    outputDict["ID"].append(rowName)
    yPredicted = getYOutputValue(yPredicted)
    outputDict["Class"].append(yPredicted)

outputDataframe = pd.DataFrame(outputDict)
outputDataframe.to_csv("file.csv", index=False)