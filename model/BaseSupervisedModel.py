from abc import abstractmethod
import pandas as pd
from tensorflow.python.keras.engine import data_adapter


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)


data_adapter._is_distributed_dataset = _is_distributed_dataset


class BaseSupervisedModel:

    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY
        self.model = None
        print("BaseSupervisedModel is called...")
        pass

    @abstractmethod
    def training(self):
        pass

    def prediction(self, testX, testY, testID):
        pre = self.model.predict(testX)
        pre = pre.flatten()
        t = pd.DataFrame({'date': testID, 'predictY': pre, 'realY': testY})
        t['predictY'] = t['predictY'].apply(lambda x: format(x, '0.2f'))
        print(t.head(20))
        t[['date', 'predictY']].to_csv("./dataset/predict", header=None, sep=" ", index=None)
        print("prediction saved")
        pass
