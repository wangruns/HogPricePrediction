from HogPricePrediction.model.DNNModel import AModel
from HogPricePrediction.utils.DataHelper import loadData, dataExpand, getNDArray
import numpy as np


if __name__ == '__main__':
    futureSwitch = True
    switchSize = 30
    dataFilePath = "./dataset/data"
    learningRate = 1e-3
    epochs = 200
    threshold = 3.1
    cost = 0x7fffffff

    df = loadData(dataFilePath)
    print(df.head())
    dataExpand(df)
    print(df.head())
    trainData, testData, dataSet = getNDArray(df, futureSwitch, switchSize)
    print(trainData[0:5])

    dnnModel = AModel(trainData[:, :-2].astype(np.float32), trainData[:, -2].astype(np.float32))
    while cost > threshold:
        if cost != 0x7fffffff:
            print("Something wrong with parameter initializing...")
        cost = dnnModel.training(learningRate, epochs)
    dnnModel.prediction(dataSet[:, :-2].astype(np.float32), dataSet[:, -2], dataSet[:, -1])
    pass
