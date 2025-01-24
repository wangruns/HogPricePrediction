from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.optimizer_v2.adam import Adam
from HogPricePrediction.model.BaseSupervisedModel import BaseSupervisedModel


class AModel(BaseSupervisedModel):
    def __init__(self, trainX, trainY):
        super(AModel, self).__init__(trainX, trainY)
        print("AModel is called...")
        pass

    def training(self, learningRate=1e-3, epochs=200):
        print("training is called...")
        self.model = Sequential([
            Dense(units=25, activation='relu'),
            Dense(units=15, activation='relu'),
            Dense(units=5, activation='relu'),
            Dense(units=1, activation='relu'),
        ])
        self.model.compile(optimizer=Adam(learning_rate=learningRate), loss=MeanSquaredError())
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs)
        return history.history['loss'][-1]
        pass
