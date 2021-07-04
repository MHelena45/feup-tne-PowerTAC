from http.server import BaseHTTPRequestHandler, HTTPStatus, ThreadingHTTPServer
import json

from WholesalePred.Preprocessing import Preprocessing
from WholesalePred.Preprocessing import PreprocessingClassification
from WholesalePred.Algorithms.LinearRegression import LinearRegressionClass
from WholesalePred.Algorithms.RandomForestRegression import RandomForestRegressionClass
from WholesalePred.Algorithms.RandomForestClassification import RandomForestClassificationClass
from WholesalePred.Algorithms.NeuralNetwork import NeuralNetworkClass
from WholesalePred.Algorithms.NeuralNetworkRegression import NeuralNetworkRegressionClass
from WholesalePred.Algorithms.NeuralNetworkClassification import NeuralNetworkClassificationClass
from WholesalePred.Model import Model
from WholesalePred.NoPrice import NoPrice

# metadata_model = [('RandomForestRegression', RandomForestRegressionClass), ('NeuralNetworkRegression', NeuralNetworkClass), ('LinearRegression', LinearRegressionClass)] 
# metadata_model_classification = [('NeuralNetworkClassification',NeuralNetworkClassificationClass), ('RandomForestClassification', RandomForestClassificationClass)] 

metadata_model = [('NeuralNetworkRegressionSK', NeuralNetworkRegressionClass)] 
metadata_model_classification = [('RandomForestClassification', RandomForestClassificationClass)] 

trainRegression = False
trainClassification = False

models = []
models_classification = []

for model in metadata_model:
    if trainRegression == True:
        model = Model(model[0], model[1]() )
        model.train_csv('WholesalePred/data.csv')
        models.append( model )
    else: models.append(Model(model[0], Model.load_model(model[0])))        

for model_classification in metadata_model_classification:
    if trainClassification == True:
        model_classification = Model(model_classification[0], model_classification[1]() )
        model_classification.train_csv('WholesalePred/dataClassification.csv')
        models_classification.append( model_classification )
    else: models_classification.append(Model(model_classification[0], Model.load_model(model_classification[0])))
        

print('Server is ready!')

class Server:
    class OurBaseHandler(BaseHTTPRequestHandler):
        def _set_OK_response(self):
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

        def do_GET(self):
            self._set_OK_response()
            self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            json_string = post_data.decode('utf-8')

            '''
            print(f"""POST request,
            Headers:
            {str(self.headers)}
            """)
            '''

            data_dict = json.loads(json_string)

            try:
                # get Classification data
                X_list_classification, y_list_classification = PreprocessingClassification.format_transform(data_dict)
                for model_classification in models_classification :
                    prediction = model_classification.sample_predict(X_list_classification)
                    # if model_classification.get_name() == 'NeuralNetworkRegression':
                    model_classification.get_error(y_list_classification, prediction)
                    model_classification.get_total_error()
                    model_classification.sample_train(X_list_classification, y_list_classification)

            except Exception:
                print("Prediction was not binary")
                pass

            try:
                X_list, y_list = Preprocessing.format_transform(data_dict)
                for model in models:
                    prediction = model.sample_predict(X_list)
                    # if model.get_name() == 'RandomForestRegression':
                    model.get_error(y_list, prediction)
                    model.get_total_error()
                    model.sample_train(X_list, y_list)

            except NoPrice as _:
                print("No trades happened at this timeslot")

            self._set_OK_response()
            self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

    @staticmethod
    def serve_endpoint(address="localhost", port=4443):
        server_address = (address, port)
        server = ThreadingHTTPServer(server_address, Server.OurBaseHandler)

        server.serve_forever()
