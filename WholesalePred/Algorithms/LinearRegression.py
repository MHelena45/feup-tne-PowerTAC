import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class LinearRegressionClass:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, data_X, data_Y):
        self.model.fit(data_X, data_Y)

    def predict(self, data):
        return self.model.predict(data)

    def train_csv(self, file_path):
        dataset = pd.read_csv(file_path)
    
        # Preparing the Data
        X = dataset[['number_competitors','number_customers','temperature','cloud_cover','wind_direction','wind_speed',
                    'temperature1','cloudCover1','windDirection1','windSpeed1','temperature2','cloudCover2','windDirection2',
                    'windSpeed2','temperature3','cloudCover3','windDirection3','windSpeed3','temperature4','cloudCover4','windDirection4',
                    'windSpeed4','temperature5','cloudCover5','windDirection5','windSpeed5','temperature6','cloudCover6','windDirection6',
                    'windSpeed6','temperature7','cloudCover7','windDirection7','windSpeed7','temperature8','cloudCover8','windDirection8',
                    'windSpeed8','temperature9','cloudCover9','windDirection9','windSpeed9','temperature10','cloudCover10','windDirection10',
                    'windSpeed10','temperature11','cloudCover11','windDirection11','windSpeed11','temperature12','cloudCover12','windDirection12',
                    'windSpeed12','temperature13','cloudCover13','windDirection13','windSpeed13','temperature14','cloudCover14','windDirection14',
                    'windSpeed14','temperature15','cloudCover15','windDirection15','windSpeed15','temperature16','cloudCover16','windDirection16',
                    'windSpeed16','temperature17','cloudCover17','windDirection17','windSpeed17','temperature18','cloudCover18','windDirection18',
                    'windSpeed18','temperature19','cloudCover19','windDirection19','windSpeed19','temperature20','cloudCover20','windDirection20',
                    'windSpeed20','temperature21','cloudCover21','windDirection21','windSpeed21','temperature22','cloudCover22','windDirection22',
                    'windSpeed22','temperature23','cloudCover23','windDirection23','windSpeed23','temperature0','cloudCover0','windDirection0',
                    'windSpeed0']]
        y = dataset['clearingPrice']

        self.train(X, y)

    def get_total_error(self, real_value, prediction_value, meanAbsoluteError, meanSquaredError, rootMeanSquaredError):
        print('Linear Regression:')
        
        # Evaluating the Algorithm

        print('Mean Absolute Error:', metrics.mean_absolute_error(real_value, prediction_value))
        print('Mean Squared Error:', metrics.mean_squared_error(real_value, prediction_value))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(real_value, prediction_value)))

        timeSlot = len(meanAbsoluteError)
        t = list(range(timeSlot))
        
        if timeSlot == 101 or timeSlot == 151 or timeSlot == 201 or timeSlot == 251:
            plt.plot(t, meanAbsoluteError, 'r--', t, rootMeanSquaredError, 'b^')
            # plt.plot(t, meanAbsoluteError, 'r--', t, meanSquaredError, 'bs', t, rootMeanSquaredError, 'g^')

            plt.xlabel("Time slot")
            plt.ylabel("Errors")
            plt.title("Linear Regression Error")
            red_patch = mpatches.Patch(color='red', label='Mean Absolute Error')
            blue_line = mlines.Line2D([], [], color='blue', marker='^', markersize=15, label='Root Mean Squared Error')
            plt.legend(handles=[red_patch,blue_line])
            plt.savefig("WholesalePred/plots/Regression/LinearRegression_Error_" + str(timeSlot) + "Timeslots_with_legend.png")  

            

    def get_error(self, real_value, prediction_value):
        print('\nLinear Regression:')
        print('Predicted value: ', prediction_value[0], '   Real value: ', real_value[0])
        
        # Evaluating the Algorithm

        singleMeanAbsoluteError = metrics.mean_absolute_error(real_value, prediction_value)
        print('Mean Absolute Error:', singleMeanAbsoluteError)

        singleMeanSquaredError = metrics.mean_squared_error(real_value, prediction_value)
        print('Mean Squared Error:', singleMeanSquaredError)

        singleRootMeanSquaredError = np.sqrt(metrics.mean_squared_error(real_value, prediction_value))
        print('Root Mean Squared Error:', singleRootMeanSquaredError)

        return singleMeanAbsoluteError, singleMeanSquaredError, singleRootMeanSquaredError  
        