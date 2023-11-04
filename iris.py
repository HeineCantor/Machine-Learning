import pandas as pd
from scipy.io.arff import loadarff
from torch.utils.data import Dataset, random_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PERCENTAGE = 0.8
TEST_PERCENTAGE = 0.2

class IrisDataset(Dataset):
    def __init__(self, dataFrame):
        self.dataframe = dataFrame

    def __getitem__(self, index):
        row = self.dataframe.iloc[index].to_numpy()
        features = row[0:-1]
        label = row[-1]
        return features, label

    def __len__(self):
        return len(self.dataframe)

raw_data = loadarff('./iris.arff')
dataFrame = pd.DataFrame(raw_data[0])

irisDataset = IrisDataset(dataFrame)

trainDataSamples = int(TRAIN_PERCENTAGE * len(irisDataset))
testDataSamples = len(irisDataset) - trainDataSamples

train_data, test_data = random_split(irisDataset, [trainDataSamples, testDataSamples])

print(f"TRAINING DATA LENGTH: {len(train_data)}")
print(f"TEST DATA LENGTH: {len(test_data)}")

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

trainSamples = []
trainLabels = []

for sample, label in train_data:
    trainSamples.append(sample)
    trainLabels.append(label)

clf.fit(trainSamples, trainLabels)

accuracy = 0

for sample, label in test_data:
    guess = clf.predict([sample])
    if guess == label:
        accuracy += 1

accuracy /= len(test_data)

print(f"ACCURACY: {accuracy}")