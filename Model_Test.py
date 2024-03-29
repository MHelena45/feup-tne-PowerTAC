import pandas as pd
from sklearn.model_selection import train_test_split
import os
from WholesalePred.Model import Model
from sklearn.preprocessing import StandardScaler
from WholesalePred.Algorithms.RandomForestRegression import RandomForestRegressionClass
from WholesalePred.Algorithms.RandomForestClassification import RandomForestClassificationClass
from WholesalePred.Algorithms.LinearRegression import LinearRegressionClass
from WholesalePred.Algorithms.NeuralNetwork import NeuralNetworkClass
from WholesalePred.Algorithms.NeuralNetworkClassification import NeuralNetworkClassificationClass
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

def getDataset():
    # get relative path
    dir = os.path.dirname(__file__)
    dataset = pd.read_csv(os.path.join(dir,'WholesalePred/data.csv'))
    dataset.head()
    return dataset

def LinearRegression():
    model = Model("LinearRegression", LinearRegressionClass())
    dataset = getDataset()
    
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



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model.sample_train(X_train, y_train)
    # model.sample_train([[9.00,3571,1976,0.5250]], [[541]])

    y_pred = model.sample_predict(X_test)
    model.get_error(y_test, y_pred)
    
    model.train_csv('WholesalePred/data.csv')

    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])
    model.get_error([18.16321023355961], prediction)
    

def LinearRegression1():
    model = Model("LinearRegression", LinearRegressionClass())
    model.train_csv('WholesalePred/data.csv')

    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])    
    print(prediction)
    model.get_error([18.16321023355961], prediction)

def RandomForestRegression():
    model = Model("RandomForestRegression", RandomForestRegressionClass())
    dataset = getDataset()

    # Preparing Data For Training - geting the right columns
    X = dataset.iloc[:, 0:102].values # not inclusivé [0,102[
    y = dataset.iloc[:, 102].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.3, test_size=0.1, random_state=0)

    # Feature Scaling
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    model.sample_train(X_train, y_train)
    # model.sample_train([[9.00,3571,1976,0.5250]], [541])
    model.sample_train([[2,203,6.2,1,160,7,6.4,1,164,5.25,6.4,0.998,168,6.46,6.6,1,154,5.57,7.3,1,159,5.61,7,1,155,4.58,8,1,167,6.31,8.4,1,167,6.47,8.4,1,175,6.18,8.9,1,173,6.13,9.7,1,167,6.33,9.7,1,170,5.5,10.9,1,163,6.31,11.8,1,183,8.07,12.1,1,193,6.6,11.5,1,204,7.48,11.3,1,208,6.39,12,1,219,7.68,12,1,223,6.71,11.8,1,221,5.56,10.7,1,196,1.42,10,1,205,5.7,9.5,1,216,4.6,9.7,1,227,2.54,9.7,1,234,1.62]],[18.16321023355961])


    y_pred = model.sample_predict(X_test)
    
    model.get_error(y_test,y_pred)

def RandomForestRegression1():
    model = Model("RandomForestRegression", RandomForestRegressionClass())
    
    model.train_csv('WholesalePred/data.csv')
    print('Model trained')

    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])    
    print(prediction)
    model.get_error([18.16321023355961], prediction)

def RandomForestClassification():
    model = Model("RandomForestClassification", RandomForestClassificationClass())
    dir = os.path.dirname(__file__)
    dataset = pd.read_csv(os.path.join(dir,'WholesalePred/dataClassification.csv'))


    # Preparing Data For Training - geting the right columns
    X = dataset.iloc[:, 0:102].values # not inclusivé [0,102[
    y = dataset.iloc[:, 102].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.6, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model.sample_train(X_train, y_train)
    model.sample_train([[2,203,14.4,0,170,3,14.4,0.873,182,4.45,16.8,1,183,4.02,17.1,1,197,3.26,15.7,1,194,4.67,16,0.983,198,4.48,14.9,1,193,3.44,13.7,1,177,3.66,13.4,1,165,2.5,14.9,1,147,1.36,12.5,0.22,147,3.27,13.3,0.764,143,3.95,14.5,0.978,149,3.98,13.8,0.871,171,3.84,15,0.987,162,4.89,14,0.993,160,3.82,13.5,1,158,2.76,13.9,1,145,4.01,13.8,0.967,152,4.07,13,0.46,162,4.12,13.2,0.964,151,4.8,13.2,0.077,163,4.86,13.6,0.561,164,5.25,14.2,0.783,175,4.13,15.4,0,187,4.13]], [1])

    y_pred = model.sample_predict(X_test)
    model.get_error(y_test,y_pred)

    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])    
    print(prediction)
    model.get_error([1], prediction)

def NeuralNetwork():
    model = Model("NeuralNetwork", NeuralNetworkClass())
    
    model.train_csv('WholesalePred/data.csv')
    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])    
    model.get_error([18.16321023355961], prediction)
    model.sample_train([[2,203,6.2,1,160,7,6.4,1,164,5.25,6.4,0.998,168,6.46,6.6,1,154,5.57,7.3,1,159,5.61,7,1,155,4.58,8,1,167,6.31,8.4,1,167,6.47,8.4,1,175,6.18,8.9,1,173,6.13,9.7,1,167,6.33,9.7,1,170,5.5,10.9,1,163,6.31,11.8,1,183,8.07,12.1,1,193,6.6,11.5,1,204,7.48,11.3,1,208,6.39,12,1,219,7.68,12,1,223,6.71,11.8,1,221,5.56,10.7,1,196,1.42,10,1,205,5.7,9.5,1,216,4.6,9.7,1,227,2.54,9.7,1,234,1.62]],[18.16321023355961])

def NeuralNetworkClassification():
    # Location of dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Assign colum names to the dataset
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=names)
    irisdata.head()

    # Assign data from first four columns to X variable
    X = irisdata.iloc[:, 0:4]
    # Assign data from first fifth columns to y variable
    y = irisdata.select_dtypes(include=[object])
    y.head()
    le = preprocessing.LabelEncoder()
    y = y.apply(le.fit_transform)
    y.Class.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    predictions = mlp.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))

def NeuralNetworkClassification1():
    model = Model("NeuralNetworkClassification", NeuralNetworkClassificationClass())
    
    model.train_csv('WholesalePred/dataClassification.csv')
    print('Model trained')

    prediction = model.sample_predict([[2,203,6.3,1,160,6,6.6,1,164,6.25,6.5,0.998,158,5.46,6.7,1,154,5.57,7.7,1,159,4.61,7.2,1,165,5.58,8.5,1,167,6.31,8.7,1,177,6.47,8.7,1,175,6.18,9.5,1,173,6.13,9.7,1,167,5.33,9.7,1,160,6.5,11.3,1,183,8.31,12.2,1,193,7.07,12.1,1,203,7.6,11.7,1,204,6.48,11.2,1,218,7.39,12,1,219,6.68,12.1,1,223,5.71,11.1,1,201,1.56,9.8,1,206,5.42,9.7,1,215,4.7,9.3,1,226,2.6,9.6,1,237,1.54,8.7,1,234,5.62]])    
    print(prediction)
    model.get_error([1], prediction)



# LinearRegression()
# LinearRegression1()
# RandomForestRegression()
# RandomForestRegression1()
# RandomForestClassification()
# NeuralNetworkClassification1()
NeuralNetwork()

