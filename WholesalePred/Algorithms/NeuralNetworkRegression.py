import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from sklearn.metrics import mean_squared_error, mean_absolute_error


class NeuralNetworkRegressionClass:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

    def train(self, data_X, data_Y):
        self.model.fit(data_X, data_Y)
 
    def predict(self, data):
        # TODO test with and without the following line
        # data = StandardScaler().fit(data)
        return self.model.predict(data)

    def train_csv(self, file_path):
        dataset = pd.read_csv(file_path)

        X = dataset.iloc[:, 0:102].values
        y = dataset.iloc[:, 102].values

        scaler = StandardScaler()
        scaler.fit(X)

        X_train = scaler.transform(X)

        self.train(X_train, y)

    def get_total_error(self, real_value, prediction_value, meanAbsoluteError, meanSquaredError, rootMeanSquaredError):
        print('Neural Network error:')
        # Evaluating the Algorithm

        print('Mean Absolute Error:', mean_absolute_error(real_value, prediction_value))
        print('Mean Squared Error:', mean_squared_error(real_value, prediction_value))
        print('Root Mean Squared Error:', np.sqrt(mean_squared_error(real_value, prediction_value)))

        timeSlot = len(meanAbsoluteError)
        t = list(range(timeSlot))
        
        if timeSlot == 100 or timeSlot == 150 or timeSlot == 200 or timeSlot == 250:
            plt.plot(t, meanAbsoluteError, 'r--', t, rootMeanSquaredError, 'b^')
            # plt.plot(t, meanAbsoluteError, 'r--', t, meanSquaredError, 'bs', t, rootMeanSquaredError, 'g^')

            red_patch = mpatches.Patch(color='red', label='Mean Absolute Error')
            blue_line = mlines.Line2D([], [], color='blue', marker='^', markersize=15, label='Root Mean Squared Error')
            plt.legend(handles=[red_patch,blue_line])

            plt.xlabel("Time slot")
            plt.ylabel("Errors")
            plt.title("Neural Network_Error")
            plt.savefig("WholesalePred/plots/Regression/NeuralNetworkRegression_Error" + str(timeSlot) + "timeslot_with_legend.png")  


    def get_error(self, real_value, prediction_value):
        print('\nNeural Network Regression:')
        print('Predicted value: ', prediction_value[0], '   Real value: ', real_value[0])
        
        # Evaluating the Algorithm

        singleMeanAbsoluteError = mean_absolute_error(real_value, prediction_value)
        print('Mean Absolute Error:', singleMeanAbsoluteError)

        singleMeanSquaredError = mean_squared_error(real_value, prediction_value)
        print('Mean Squared Error:', singleMeanSquaredError)

        singleRootMeanSquaredError = np.sqrt(mean_squared_error(real_value, prediction_value))
        print('Root Mean Squared Error:', singleRootMeanSquaredError)

        return singleMeanAbsoluteError, singleMeanSquaredError, singleRootMeanSquaredError 
