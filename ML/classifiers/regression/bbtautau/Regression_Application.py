from __future__ import division
import theano
import numpy as np
import pandas
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, GaussianNoise, BatchNormalization, Merge
from keras.layers.advanced_activations import PReLU
from keras.models import model_from_json
import theano.tensor as T
from sklearn.pipeline import Pipeline
import json
import glob
import pickle
from sklearn.externals import joblib
from abc import ABCMeta, abstractmethod

class Regressor(object):
    __metaclass__ = ABCMeta

    def _loadRegressor(self, name):
        with open(name + '_compile.json', 'r') as fin:
                                    self.compileArgs = json.load(fin)
        for i in range(len(glob.glob(name + '*.h5'))):
            model = model_from_json(open(name + '_' + str(i) + '.json').read())
            model.load_weights(name + "_" + str(i) + '.h5')
            model.compile(**self.compileArgs)
            self.ensemble.append(model)
        print(len(self.ensemble), "components found in ensemble")
        with open(name + '_weights.pkl', 'rb') as fin:
            self.weights = pickle.load(fin)
        with open(name + '_inputPipe.pkl', 'rb') as fin:
            self.inputPipe = pickle.load(fin)
        with open(name + '_outputPipe.pkl', 'rb') as fin:
            self.outputPipe = pickle.load(fin)
        
    def evalResponse(self):
        pred = np.zeros((len(self.data), len(self.outputFeatures)))
        for i, model in enumerate(self.ensemble):
            pred += self.weights[i]*self.outputPipe.inverse_transform(model.predict(self.inputPipe.transform(self.data[self.inputFeatures].values.astype(theano.config.floatX)), verbose=0))
        for n, feature in enumerate(self.outputFeatures):
            self.data[feature] = pandas.Series(pred[:,n], index=self.data.index)

    def getExtraVariables(self):
        prefix0 = self._regName + self._objects[0] + "_"
        prefix1 = self._regName + self._objects[1] + "_"
        prefixC = self._regName + self._combinedObject + "_"
        self.data[prefix0 + '|p|'] = np.sqrt(np.square(self.data.loc[:, prefix0 + 'px'])+np.square(self.data.loc[:, prefix0 + 'py'])+np.square(self.data.loc[:, prefix0 + 'pz']))
        self.data[prefix0 + 'E'] = np.sqrt(np.square(self._mass)+np.square(self.data.loc[:, prefix0 + '|p|']))
        self.data[prefix1 + '|p|'] = np.sqrt(np.square(self.data.loc[:, prefix1 + 'px'])+np.square(self.data.loc[:, prefix1 + 'py'])+np.square(self.data.loc[:, prefix1 + 'pz']))
        self.data[prefix1 + 'E'] = np.sqrt(np.square(self._mass)+np.square(self.data.loc[:, prefix1 + '|p|']))
        self.data[prefixC + 'px'] = self.data.loc[:, prefix0 + 'px']+self.data.loc[:, prefix1 + 'px']
        self.data[prefixC + 'py'] = self.data.loc[:, prefix0 + 'py']+self.data.loc[:, prefix1 + 'py']
        self.data[prefixC + 'pz'] = self.data.loc[:, prefix0 + 'pz']+self.data.loc[:, prefix1 + 'pz']
        self.data[prefixC + 'E'] = self.data.loc[:, prefix0 + 'E']+self.data.loc[:, prefix1 + 'E']
        self.data[prefixC + '|p|'] = np.sqrt(np.square(self.data.loc[:, prefixC + 'px'])+np.square(self.data.loc[:, prefixC + 'py'])+np.square(self.data.loc[:, prefixC + 'pz']))
        self.data[prefixC + 'mass'] = np.sqrt(np.square(self.data.loc[:, prefixC + 'E'])-np.square(self.data.loc[:, prefixC + '|p|']))
        
    def refineDiHiggsVector(self):
        prefixC = self._regName + "diH_"
        prefixB = self._prefixB
        prefixT = self._prefixT
        self.data[prefixC + 'px'] = self.data.loc[:, prefixB + 'px']+self.data.loc[:, prefixT + 'px']
        self.data[prefixC + 'py'] = self.data.loc[:, prefixB + 'py']+self.data.loc[:, prefixT + 'py']
        self.data[prefixC + 'pz'] = self.data.loc[:, prefixB + 'pz']+self.data.loc[:, prefixT + 'pz']
        self.data[prefixC + 'E'] = self.data.loc[:, prefixB + 'E']+self.data.loc[:, prefixT + 'E']
        self.data[prefixC + '|p|'] = np.sqrt(np.square(self.data.loc[:, prefixC + 'px'])+np.square(self.data.loc[:, prefixC + 'py'])+np.square(self.data.loc[:, prefixC + 'pz']))
        self.data[prefixC + 'mass'] = np.sqrt(np.square(self.data.loc[:, prefixC + 'E'])-np.square(self.data.loc[:, prefixC + '|p|']))   

class BPairRegressor(Regressor):    
    
    def __init__(self, inData, name, mode):
        self._regName = "regB_"
        self.outputFeatures = [self._regName + x for x in ['b_0_px', 'b_0_py', 'b_0_pz', 'b_1_px', 'b_1_py', 'b_1_pz']]
        self._objects = ["b_0", "b_1"]
        self._combinedObject = "h_bb"
        self._mass = 4.8
        self._prefixB = 'regB_h_bb_'
        self._prefixT = 'h_tt_'
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.outputPipe = None
        self.compileArgs = None
        if mode == "mu_tau_b_b":
            self.inputFeatures = ['t_0_mass', 't_1_mass', 'b_0_mass', 'b_1_mass', 'mT', 'hT', 'sT', 'centrality', 'eVis', 't_0_px', 't_0_py', 't_0_pz', 't_0_|p|', 't_0_E', 't_1_px', 't_1_py', 't_1_pz', 't_1_|p|', 't_1_E', 'b_0_px', 'b_0_py', 'b_0_pz', 'b_0_|p|', 'b_0_E', 'b_1_px', 'b_1_py', 'b_1_pz', 'b_1_|p|', 'b_1_E', 'mPT_px', 'mPT_py', 'mPT_|p|', 't_1_mT', 'h_tt_mass', 'h_bb_mass', 'diH_mass', 'h_tt_px', 'h_tt_py', 'h_tt_pz', 'h_tt_|p|', 'h_tt_E', 'h_bb_px', 'h_bb_py', 'h_bb_pz', 'h_bb_|p|', 'h_bb_E', 'diH_px', 'diH_py', 'diH_pz', 'diH_|p|', 'diH_E', 'hl_mT']
        self.data = inData
        self._loadRegressor(name)

class TauPairRegressor(Regressor):
    
    def __init__(self, inData, name, mode):
        self._regName = "regTau_"
        self.outputFeatures = [self._regName + x for x in ['t_0_px', 't_0_py', 't_0_pz', 't_1_px', 't_1_py', 't_1_pz']]
        self._objects = ["t_0", "t_1"]
        self._combinedObject = "h_tt"
        self._mass = 1.77686
        self._prefixB = 'regB_h_bb_'
        self._prefixT = 'regTau_h_tt_'
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.outputPipe = None
        self.compileArgs = None
        if mode == "mu_tau_b_b":
            self.inputFeatures = ['t_0_mass', 't_1_mass', 'b_0_mass', 'b_1_mass', 'mT', 'hT', 'sT', 'centrality', 'eVis', 't_0_px', 't_0_py', 't_0_pz', 't_0_|p|', 't_0_E', 't_1_px', 't_1_py', 't_1_pz', 't_1_|p|', 't_1_E', 'mPT_px', 'mPT_py', 'mPT_|p|', 't_1_mT', 'regB_b_0_px', 'regB_b_0_py', 'regB_b_0_pz', 'regB_b_1_px', 'regB_b_1_py', 'regB_b_1_pz', 'regB_b_0_|p|', 'regB_b_0_E', 'regB_b_1_|p|', 'regB_b_1_E', 'regB_h_bb_px', 'regB_h_bb_py', 'regB_h_bb_pz', 'regB_h_bb_E', 'regB_h_bb_|p|', 'regB_h_bb_mass', 'regB_diH_px', 'regB_diH_py', 'regB_diH_pz', 'regB_diH_E', 'regB_diH_|p|', 'regB_diH_mass', 'h_tt_mass', 'h_tt_px', 'h_tt_py', 'h_tt_pz', 'h_tt_|p|', 'h_tt_E', 'hl_mT']
        self.data = inData
        self._loadRegressor(name)
        
class HHMomRegressor(Regressor):

    refineDiHiggsVector = None
    
    def getExtraVariables(self):
        prefix = self._regName + "diH_"
        self.data[prefix + '|p|'] = np.sqrt(np.square(self.data.loc[:, prefix + 'px'])+np.square(self.data.loc[:, prefix + 'py'])+np.square(self.data.loc[:, prefix + 'pz']))
        self.data[prefix + 'E'] = np.sqrt(np.square(self.data.loc[:, next(x for x in self.inputFeatures if "diH_mass" in x)])+np.square(self.data.loc[:, prefix + '|p|']))
    
    def __init__(self, inData, name, mode):
        self._regName = "regHH_"
        self.outputFeatures = [self._regName + x for x in ['diH_px', 'diH_py', 'diH_pz']]
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.outputPipe = None
        self.compileArgs = None
        if mode == "mu_tau_b_b":
            self.inputFeatures = ['t_0_mass', 'b_0_mass', 'b_1_mass', 'mT', 'hT', 'sT', 'centrality', 'eVis', 'mPT_px', 'mPT_py', 'mPT_|p|', 't_1_mT', 'regB_b_0_px', 'regB_b_0_py', 'regB_b_0_pz', 'regB_b_1_px', 'regB_b_1_py', 'regB_b_1_pz', 'regB_b_0_|p|', 'regB_b_0_E', 'regB_b_1_|p|', 'regB_b_1_E', 'regB_h_bb_px', 'regB_h_bb_py', 'regB_h_bb_pz', 'regB_h_bb_E', 'regB_h_bb_|p|', 'regB_h_bb_mass', 'regTau_t_0_px', 'regTau_t_0_py', 'regTau_t_0_pz', 'regTau_t_1_px', 'regTau_t_1_py', 'regTau_t_1_pz', 'regTau_t_0_|p|', 'regTau_t_0_E', 'regTau_t_1_|p|', 'regTau_t_1_E', 'regTau_h_tt_px', 'regTau_h_tt_py', 'regTau_h_tt_pz', 'regTau_h_tt_E', 'regTau_h_tt_|p|', 'regTau_h_tt_mass', 'regTau_diH_px', 'regTau_diH_py', 'regTau_diH_pz', 'regTau_diH_E', 'regTau_diH_|p|', 'regTau_diH_mass', 'hl_mT']
        self.data = inData
        self._loadRegressor(name)
    
class HHRegressor(Regressor):

    refineDiHiggsVector = None
    
    def getExtraVariables(self):
        prefix = self._regName + "diH_"
        self.data[prefix + 'E'] = np.sqrt(np.square(self.data.loc[:, 'regHH_diH_mass'])+np.square(self.data.loc[:, prefix + '|p|']))
    
    def __init__(self, inData, name, mode):
        self._regName = "regHH_"
        self.outputFeatures = [self._regName + x for x in ['diH_mass']]
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.outputPipe = None
        self.compileArgs = None
        if mode == "mu_tau_b_b":
            self.inputFeatures = ['b_0_mass', 'b_1_mass', 't_0_mass', 't_1_mass', 'mPT_px', 'mPT_py', 'regB_b_0_px',
                                  'regB_b_0_py', 'regB_b_0_pz', 'regB_b_1_px', 'regB_b_1_py', 'regB_b_1_pz', 'regB_b_0_|p|',
                                  'regB_b_0_E', 'regB_b_1_|p|', 'regB_b_1_E', 'regB_h_bb_px', 'regB_h_bb_py', 'regB_h_bb_pz',
                                  'regB_h_bb_E', 'regB_h_bb_|p|', 'regB_h_bb_mass', 'regTau_t_0_px', 'regTau_t_0_py',
                                  'regTau_t_0_pz', 'regTau_t_1_px', 'regTau_t_1_py', 'regTau_t_1_pz', 'regTau_t_0_|p|',
                                  'regTau_t_0_E', 'regTau_t_1_|p|', 'regTau_t_1_E', 'regTau_h_tt_px', 'regTau_h_tt_py',
                                  'regTau_h_tt_pz', 'regTau_h_tt_E', 'regTau_h_tt_|p|', 'regTau_h_tt_mass', 'regHH_diH_px',
                                  'regHH_diH_py', 'regHH_diH_pz', 'regHH_diH_|p|', 'regHH_diH_E', 'hl_mT', 'regTau_diH_mass']
        self.data = inData
        self._loadRegressor(name)
        
class HBBMassRegressor(Regressor):

    refineDiHiggsVector = None

    def _setNames(self):
        self.inputFeatures0 = list(self._inputFeatures)
        self.inputFeatures1 = list(self._inputFeatures)
        for i in range(len(self.inputFeatures1 )):
            if "b_0" in self.inputFeatures1 [i]:
                self.inputFeatures1 [i] = self.inputFeatures1 [i][0:self.inputFeatures1 [i].find("b_0")] + "b_1" + self.inputFeatures1 [i][self.inputFeatures1 [i].find("b_0")+3:]
        self.outputFeatures0 = list(self._outputFeatures)
        self.outputFeatures1 = list(self._outputFeatures)
        for i in range(len(self.outputFeatures1)):
            if "b_0" in self.outputFeatures1[i]:
                self.outputFeatures1[i] = self.outputFeatures1[i][0:self.outputFeatures1[i].find("b_0")] + "b_1" + self.outputFeatures1[i][self.outputFeatures1[i].find("b_0")+3:]       
        
    def evalResponse(self):
        pred0 = np.zeros((len(self.data), len(self.outputFeatures0)))
        for i, model in enumerate(self.ensemble):
            pred0 += self.weights[i]*self.outputPipe.inverse_transform(model.predict(self.inputPipe.transform(self.data[self.inputFeatures0].values.astype(theano.config.floatX)), verbose=0))
        pred1 = np.zeros((len(self.data), len(self.outputFeatures1)))
        for i, model in enumerate(self.ensemble):
            pred1 += self.weights[i]*self.outputPipe.inverse_transform(model.predict(self.inputPipe.transform(self.data[self.inputFeatures1].values.astype(theano.config.floatX)), verbose=0))
        for n, feature in enumerate(self.outputFeatures0):
            self.data[feature] = pandas.Series(pred0[:,n], index=self.data.index)
        for n, feature in enumerate(self.outputFeatures1):
            self.data[feature] = pandas.Series(pred1[:,n], index=self.data.index)
    
    def __init__(self, inData, name, mode):
        self._regName = "regHBB_"
        self._objects = ["b_0", "b_1"]
        self._combinedObject = "h_bb"
        self._mass = 4.8
        self.ensemble = []
        self.weights = None
        self.inputPipe = None
        self.outputPipe = None
        self.compileArgs = None
        if mode == "mu_tau_b_b":
            self._inputFeatures = ['b_0_mass', 'b_0_px', 'b_0_py', 'b_0_pz', 'b_0_|p|', 'b_0_E',
                                  'mPT_px', 'mPT_py',
                                  'h_bb_E',
                                  't_0_mass', 't_0_px', 't_0_py', 't_0_pz', 't_0_|p|', 't_0_E',
                                  't_1_mass', 't_1_px', 't_1_py', 't_1_pz', 't_1_|p|', 't_1_E',
                                  'hl_mT',
                                  'h_tt_mass', 'h_tt_px', 'h_tt_py', 'h_tt_pz', 'h_tt_|p|', 'h_tt_E',
                                  'diH_E', 'diH_|p|', 'diH_mass']
        self._outputFeatures = [self._regName + x for x in ['b_0_px', 'b_0_py', 'b_0_pz']]  
        self._setNames()     
        self.data = inData
        self._loadRegressor(name)

class HTTMassRegressor(Regressor):

    refineDiHiggsVector = None

    def _loadRegressor(self, name, mode):
        with open(name + '_compile.json', 'r') as fin:
                                    self.compileArgs[mode] = json.load(fin)
        for i in range(len(glob.glob(name + '*.h5'))):
            model = model_from_json(open(name + '_' + str(i) + '.json').read())
            model.load_weights(name + "_" + str(i) + '.h5')
            model.compile(**self.compileArgs[mode])
            self.ensemble[mode].append(model)
        print(len(self.ensemble[mode]), "components found in ensemble")
        with open(name + '_weights.pkl', 'r') as fin:
            self.weights[mode] = pickle.load(fin)
        with open(name + '_inputPipe.pkl', 'r') as fin:
            self.inputPipe[mode] = pickle.load(fin)
        with open(name + '_outputPipe.pkl', 'r') as fin:
            self.outputPipe[mode] = pickle.load(fin)

    def evalResponse(self):
        pred0 = np.zeros((len(self.data), len(self.outputFeatures0)))
        for i, model in enumerate(self.ensemble[0]):
            pred0 += self.weights[0][i]*self.outputPipe[0].inverse_transform(model.predict(self.inputPipe[0].transform(self.data[self.inputFeatures0].values.astype(theano.config.floatX)), verbose=0))
        pred1 = np.zeros((len(self.data), len(self.outputFeatures1)))
        for i, model in enumerate(self.ensemble[1]):
            pred1 += self.weights[1][i]*self.outputPipe[1].inverse_transform(model.predict(self.inputPipe[1].transform(self.data[self.inputFeatures1].values.astype(theano.config.floatX)), verbose=0))
        for n, feature in enumerate(self.outputFeatures0):
            self.data[feature] = pandas.Series(pred0[:,n], index=self.data.index)
        for n, feature in enumerate(self.outputFeatures1):
            self.data[feature] = pandas.Series(pred1[:,n], index=self.data.index)
    
    def __init__(self, inData, name0, name1, mode):
        self._regName = "regHTT_"
        self._objects = ["t_0", "t_1"]
        self._combinedObject = "h_tt"
        self._mass = 1.77686
        self.ensemble = [[], []]
        self.weights = [None, None]
        self.inputPipe = [None, None]
        self.outputPipe = [None, None]
        self.compileArgs = [None, None]
        if mode == "mu_tau_b_b":
            self.inputFeatures0 = ['t_0_mass', 't_0_px', 't_0_py', 't_0_pz', 't_0_|p|', 't_0_E',
                                      'mPT_px', 'mPT_py',
                                      'h_tt_E',
                                      'b_0_mass', 'b_0_px', 'b_0_py', 'b_0_pz', 'b_0_|p|', 'b_0_E',
                                      'b_1_mass', 'b_1_px', 'b_1_py', 'b_1_pz', 'b_1_|p|', 'b_1_E',
                                      'hl_mT',
                                      'h_bb_mass', 'h_bb_px', 'h_bb_py', 'h_bb_pz', 'h_bb_|p|', 'h_bb_E',
                                      'diH_E', 'diH_|p|', 'diH_mass']
            self.inputFeatures1 = ['t_1_mass', 't_1_px', 't_1_py', 't_1_pz', 't_1_|p|', 't_1_E',
                                       'mPT_px', 'mPT_py',
                                       'h_tt_E',
                                       'b_0_mass', 'b_0_px', 'b_0_py', 'b_0_pz', 'b_0_|p|', 'b_0_E',
                                       'b_1_mass', 'b_1_px', 'b_1_py', 'b_1_pz', 'b_1_|p|', 'b_1_E',
                                       'hl_mT',
                                       'h_bb_mass', 'h_bb_px', 'h_bb_py', 'h_bb_pz', 'h_bb_|p|', 'h_bb_E',
                                       'diH_E', 'diH_|p|', 'diH_mass']
        self.outputFeatures0 = [self._regName + x for x in ['t_0_px', 't_0_py', 't_0_pz']]  
        self.outputFeatures1 = [self._regName + x for x in ['t_1_px', 't_1_py', 't_1_pz']]
        self.data = inData
        self._loadRegressor(name0, 0)
        self._loadRegressor(name1, 1)