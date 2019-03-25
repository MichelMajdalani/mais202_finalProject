import pandas as pd
from joblib import dump, load

''' Predictor class for outputting model result '''


class Predictor:

    def __init__(self, verbose=True):
        # store list of columns
        self.columns = ['area_mean', 'area_se', 'area_worst', 'compactness_mean', 'compactness_se',
                        'compactness_worst', 'texture_mean', 'texture_se', 'texture_worst', 'smoothness_mean', 'symmetry_mean']
        # load the trained model
        self.model = load('model/model.joblib')

    ''' Preprocess feature data '''

    def preprocess(self, features):

        return features

    ''' Read data from a post request '''

    def read_data(self, request):
        data = []
        data.append(request.form['area_mean'])
        data.append(request.form['area_se'])
        data.append(request.form['area_worst'])
        data.append(request.form['compactness_mean'])
        data.append(request.form['compactness_se'])
        data.append(request.form['compactness_worst'])
        data.append(request.form['texture_mean'])
        data.append(request.form['texture_se'])
        data.append(request.form['texture_worst'])
        data.append(request.form['smoothness_mean'])
        data.append(request.form['symmetry_mean'])

        features = pd.DataFrame([data], columns=self.columns)
        return features

    ''' Make a prediction given a set of features '''

    def predict(self, request):
        features = self.read_data(request)
        processed = self.preprocess(features)
        prediction = self.model.predict(processed)
        return prediction
