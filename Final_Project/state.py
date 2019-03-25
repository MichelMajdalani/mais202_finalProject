import pandas as pd
from joblib import dump, load

''' State Tracker class for keeping track of current states in the frontend '''


class StateTracker:

    def __init__(self, verbose=True):
        # set the initial states
        self.state = {
            'area_mean': 551.1,
            'area_se': 24.53,
            'area_worst': 686.5,
            'compactness_mean': 0.09263,
            'compactness_se': 0.02045,
            'compactness_worst': 0.2119,
            'texture_mean': 18.84,
            'texture_se': 1.108,
            'texture_worst': 25.41,
            'smoothness_mean': 0.09587,
            'symmetry_mean': 0.1792,
            'result': ''
        }

    ''' Updates the current state '''

    def updateState(self, request):
        self.state['area_mean'] = request.form['area_mean']
        self.state['area_se'] = request.form['area_se']
        self.state['area_worst'] = request.form['area_worst']
        self.state['compactness_mean'] = request.form['compactness_mean']
        self.state['compactness_se'] = request.form['compactness_se']
        self.state['compactness_worst'] = request.form['compactness_worst']
        self.state['texture_mean'] = request.form['texture_mean']
        self.state['texture_se'] = request.form['texture_se']
        self.state['texture_worst'] = request.form['texture_worst']
        self.state['smoothness_mean'] = request.form['smoothness_mean']
        self.state['symmetry_mean'] = request.form['symmetry_mean']
