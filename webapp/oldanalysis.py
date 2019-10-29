import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import tensorflow as tf


def sampleParse(excelname, save_loc):
    excel = os.path.join(save_loc, excelname)
    GFP = pd.read_excel(excel, sheet_name=0)
    UV = pd.read_excel(excel, sheet_name=1)
    UV = UV.rename(columns={x:y for x,y in zip(UV.columns,range(0,len(UV.columns)))})

    UV.loc[UV[0] == "Wavelength:"]
    Wavelength1 = UV.iloc[28:36]
    Wavelength1.columns = Wavelength1.iloc[0]
    Wavelength1 = Wavelength1.iloc[1:]
    sample1 = Wavelength1.iloc[3:6]
    sample1.drop(sample1.columns[len(sample1.columns)-1], axis=1, inplace=True)
    sample1.drop(sample1.columns[4], axis=1, inplace=True)
    sample1.drop(sample1.columns[0], axis=1, inplace=True)

    X = [int(x) for x in sample1.columns]
    Output = []
    for index, row in sample1.iterrows():
        Y = row.get_values()
        Y = [float(y) for y in Y]
        plt.plot(X, Y, label=index)
        Output.append(str(index) + ": " +  str(linregress(X, Y)))
    plt.savefig(os.path.join(save_loc, "lineplot.png"))
    return Output

class Model:
    def __init__(self, model_dir):
        #self.model = self.build_model(model_dir)
        self.model = self.build_model_regu(model_dir)

    #Build Model
    def build_model(self, model_dir):
        continuous_features = [tf.feature_column.numeric_column(str(k)) for k in FEATURES]
        model = tf.estimator.LinearClassifier(
            n_classes = 3,
            model_dir=model_dir,
            feature_columns= continuous_features)
        return model
        #Build Model
    def build_model_regu(self, model_dir):
        continuous_features = [tf.feature_column.numeric_column(str(k)) for k in FEATURES]
        model = tf.estimator.LinearClassifier(
            n_classes = 3,
            model_dir=model_dir,
            feature_columns= continuous_features,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.9,
                l2_regularization_strength=5))
        return model

    #Train Model
    def train_model(self, df_train):
        self.model.train(input_fn=get_input_fn(df_train,
                                               num_epochs=None,
                                               n_batch = 16,
                                               shuffle=False),
                         steps=128)

    #Evaluate Model
    def eval_model(self, df_test):
        self.model.evaluate(input_fn=get_input_fn(df_test,
                                                  num_epochs=None,
                                                  n_batch = 16,
                                                  shuffle=False),
                            steps=128)
    #Make Prediction with Model
    def makePrediction(self, df):
        outputDF = df.copy()
        pred_iter = self.model.predict(tf.estimator.inputs.pandas_input_fn(df, shuffle=False))
        probabilities = []
        classifier = []

        for i in pred_iter:
            probabilities.append(i['logits'])
            classifier.append(i['class_ids'])
        outputDF['probabilities'] = probabilities
        outputDF['class_id'] = classifier
        return outputDF
def get_input_fn(data_set, num_epochs=None, n_batch = 16, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({str(k): data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set['category'].values),
        batch_size=n_batch,
        num_epochs=num_epochs,
        shuffle=shuffle)

def monteCarloTrainingData(singular_list, numSets=10, dataPts=128):
    allsets = []
    mu = np.mean(singular_list)
    sigma = np.std(singular_list)
    #print(mu, sigma)
    for _ in range(numSets):
        gen = np.random.normal(mu, sigma, dataPts)
        allsets.append(gen)
    return allsets

def createTrainingDF(allsets, category):
    trainingDF = pd.DataFrame(columns=FEATURES)
    for index in range(len(allsets)):
        trainingDF[FEATURES[index]] = allsets[index]
    trainingDF["category"] = category
    return trainingDF

def generateDF(dilutions_dict, category, numSets=10):
    allDat = {}
    for key in dilutions_dict.keys():
        allDat[key] = monteCarloTrainingData(dilutions_dict[key])
    trainingDF = []
    for index in range(numSets):
        trainingDF.append(createTrainingDF([allDat[key][index] for key in allDat.keys()], category))
    return trainingDF


FEATURES = ["12.5x", "20x", "50x", "125x", "250x", "500x"]

METADATA = ["Date:", "Time:", "Measurement mode:", "Excitation wavelength:",
            "Emission wavelength:", "Excitation bandwidth:", "Emission bandwidth:",
            "Gain (Manual):", "Number of reads:", "FlashMode:", "Integration time:", "Lag time:",
            "Part of the plate:", "Target Temperature:", "Current Temperature:"]

class PipetteTutorial:
    def __init__(self, excelname, save_loc):
        excel = os.path.join(save_loc, excelname)
        self.df = pd.read_excel(excel, sheet_name=0)
        self.metadata = self.parseMetadata()
        self.data = self.parseData()
    def parseMetadata(self):
        df = self.df
        metadata = {}
        def parse(name):
            df2 = df.loc[df[df.columns[0]]==name].dropna(axis=1)
            return df2[df2.columns[1]].iloc[0]
        for item in METADATA:
            metadata[item] = parse(item)
        return metadata
    def parseData(self):
        data = {}
        df = self.df
        start_index = df.loc[df[df.columns[0]]=='<>'].index[0]
        df2 = df.iloc[start_index+1:]
        index = 'A'
        for row in df2.iterrows():
            df_row = list(row[1][1:])
            if "..." not in df_row:
                data[index] = df_row
                index = chr(ord(index) + 1)
        return pd.DataFrame(data)
    def dilutionLine(self, row, save_loc, df):
        Output = []
        print(df)
        plt.plot(df[row])
        plt.title(row)
        Output.append(str(row) + ": " +  str(linregress(df[row], df.index)))
        plt.savefig(os.path.join(save_loc, row+"_lineplot.png"))
        plt.close()
        return Output

def runAnalysis(excelname, save_loc, model_directory):
    trial = PipetteTutorial(excelname, save_loc)
    first = ord(trial.metadata["Part of the plate:"].split(" - ")[0][:1])
    last = ord(trial.metadata["Part of the plate:"].split(" - ")[1][:1])
    documentation = {}
    index = 0

    '''
    while first <= last:
        plate = chr(first)
        if index < len(FEATURES):
            documentation[str(plate + "_" + FEATURES[index])] = trial.dilutionLine(plate, save_loc, trial.data)
        first += 1
        index += 1
    '''

    model = Model(os.path.join(model_directory))



    def performTest(df):
        test_data = {}
        index = 0
        for val in df["A"].iloc[:6]:
            test_data[FEATURES[index]] = [val]
            index += 1
        return test_data

    documentation["user data"] = trial.dilutionLine("user data", save_loc, pd.DataFrame({"user data": trial.data["A"].iloc[:6]}))
    results = model.makePrediction(pd.DataFrame(performTest(trial.data)))
    return documentation, results, trial.metadata
