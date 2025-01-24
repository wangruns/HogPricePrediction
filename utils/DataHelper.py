import pandas as pd
import numpy as np
import datetime


def loadData(dataFilePath):
    print("loadData is called...")
    df = pd.read_table(dataFilePath, header=None, names=['dmy', 'price'], sep=" ")
    return df


def dataExpand(df):
    print("dataExpand is called...")
    _df = pd.concat([df, df['dmy'].str.split('/', expand=True)], axis=1)
    df['d'], df['m'], df['y'] = _df[0].astype(float), _df[1].astype(float), _df[2].astype(float)
    pass


def _featureScale(nDArray, mu=None, sigma=None):
    print("_featureScale is called...")
    if mu is None or sigma is None:
        mu = nDArray[:, :-2].mean(axis=0)
        sigma = nDArray[:, :-2].astype(np.float32).std(axis=0)
    nDArray[:, :-2] = (nDArray[:, :-2] - mu) / sigma
    return mu, sigma


def _hybridFutureData(testData, switchSize, mu, sigma, d, m, y):
    print("_hybridFutureData is called...")
    inDt = '{}/{}/{}'.format(d, m, y)
    inDt = datetime.datetime.strptime(inDt, "%d/%m/%Y")
    tempDF = pd.DataFrame(columns=['d', 'm', 'y', 'price', 'dmy'])

    for i in range(switchSize):
        row = pd.DataFrame([[inDt.day, inDt.month, inDt.year, 0x7fffffff, inDt.strftime("%d/%m/%Y")]],
                           columns=tempDF.columns)
        tempDF = pd.concat([tempDF, row], ignore_index=True)
        inDt += datetime.timedelta(days=1)
    _nDArray = np.array(tempDF[['d', 'm', 'y', 'price', 'dmy']])
    _featureScale(_nDArray, mu, sigma)

    testData = np.append(testData, _nDArray, axis=0)
    return testData


def getNDArray(df, futureSwitch=False, switchSize=7):
    print("getNDArray is called...")
    d, m, y = df['dmy'][-1:].str.split('/', expand=True).values[0]
    _nDArray = np.array(df[['d', 'm', 'y', 'price', 'dmy']])
    mu, sigma = _featureScale(_nDArray)
    np.random.shuffle(_nDArray)
    ratio = 0.8
    trainSize = int(ratio * len(_nDArray))
    trainData = _nDArray[:trainSize]
    testData = _nDArray[trainSize:]
    if futureSwitch:
        _nDArray = _hybridFutureData(_nDArray, switchSize, mu, sigma, d, m, y)
    return trainData, testData, _nDArray
