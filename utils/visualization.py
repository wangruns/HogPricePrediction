import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
matplotlib.rc("font", family='MicroSoft YaHei', weight="bold")


def visualization():
    realDF = pd.read_table("../dataset/data", header=None, names=['dmy', 'price'], sep=" ")
    predictDF = pd.read_table("../dataset/predict", header=None, names=['dmy', 'price'], sep=" ")

    realDF['dmy'] = pd.to_datetime(realDF['dmy'], format="%d/%m/%Y")
    realDF.sort_values(by='dmy', inplace=True)

    predictDF['dmy'] = pd.to_datetime(predictDF['dmy'], format="%d/%m/%Y")
    predictDF.sort_values(by='dmy', inplace=True)

    plt.figure(figsize=(75, 25), dpi=100)
    plt.title("全国外三元生猪价格走势图", fontsize=40, weight="bold")
    plt.xlabel("日期", fontsize=20, weight="bold")
    plt.ylabel("价格", fontsize=20, weight="bold")
    plt.plot(realDF['dmy'], realDF['price'], '-g', linewidth=5.0, label="市场真实价")
    plt.plot(predictDF['dmy'], predictDF['price'], '-r', linewidth=5.0, label="模型预测价")
    plt.legend()
    plt.show()
    pass


if __name__ == '__main__':
    visualization()
