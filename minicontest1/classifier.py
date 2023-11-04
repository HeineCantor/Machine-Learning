import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim

VALIDATION_PERCENTAGE = 0.1
TRAINING_PERCENTAGE = 1 - VALIDATION_PERCENTAGE

EPOCHS = 300
BATCH_SIZE = 1

TRAIN_PATH = "~/Desktop/git/Machine Learning/minicontest1/train.csv"
TEST_PATH = "~/Desktop/git/Machine Learning/minicontest1/test.csv"

class SoilClassifier(nn.Module):
    def __init__(self, inFeatures):
        super().__init__()

        self.input = nn.Linear(inFeatures, 4)
        self.activation1 = nn.Sigmoid()
        self.hidden1 = nn.Linear(4, 8)
        self.activation2 = nn.Sigmoid()
        self.hidden2 = nn.Linear(8, 8)
        self.activation3 = nn.Sigmoid()
        self.output = nn.Linear(8, 3)
        self.activationOutput = nn.Sigmoid()
    def forward(self, x):
        x = self.activation1(self.input(x))
        x = self.activation2(self.hidden1(x))
        x = self.activation3(self.hidden2(x))
        x = self.activationOutput(self.output(x))

        return x

class SoilDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataframe = dataFrame

    def __getitem__(self, index):
        row = self.dataframe.iloc[index][1:].to_numpy(dtype="float32")
        features = row[0:-1]
        label = row[-1]
        return features, label

    def __len__(self):
        return len(self.dataframe)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

trainingDataFrame = pd.read_csv(TRAIN_PATH)
testDataFrame = pd.read_csv(TEST_PATH)

trainDataset = SoilDataset(trainingDataFrame)
testDataset = SoilDataset(testDataFrame)

trainLength = int(TRAINING_PERCENTAGE * len(trainDataset))
validationLength = len(trainDataset) - trainLength

trainDataset, validationDataset = random_split(trainDataset, [trainLength, validationLength])

model = SoilClassifier(16)
print(f"Model for training: {model}")

lossFunction = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.03)

def getYTensor(value):
    if(value == 1):
        return torch.tensor([1, 0, 0], dtype=torch.float32)
    elif(value == 2):
        return torch.tensor([0, 1, 0], dtype=torch.float32)
    else:
        return torch.tensor([0, 0, 1], dtype=torch.float32)

for epoch in range(EPOCHS):
    validationLoss = 0
    for i in range(0, len(trainDataset)):
        trainingBatch = trainDataset[i]
        yPredicted = model(torch.from_numpy(trainingBatch[0]))
        #print(yPredicted)
        yBatch = getYTensor(trainingBatch[1])
        loss = lossFunction(yPredicted, yBatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    for i in range(0, len(validationDataset)):
        validationBatch = validationDataset[i]
        validationYPredicted = model(torch.from_numpy(validationBatch[0]))
        validationLoss += lossFunction(validationYPredicted, yBatch)
    print(f"Epoch {epoch} finished. Train Loss = {loss} - Validation Loss = {validationLoss / len(validationDataset)}")

for sample in testDataset:
    yPredicted = model(torch.from_numpy(sample[0]))