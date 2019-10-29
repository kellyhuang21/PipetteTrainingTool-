import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats import logistic
import os
import pickle

FEATURES = ["1x", "2x", "4x", "8x", "16x"]

METADATA = ["Date:", "Time:", "Measurement mode:", "Excitation wavelength:",
            "Emission wavelength:", "Excitation bandwidth:", "Emission bandwidth:",
            "Gain (Manual):", "Number of reads:", "FlashMode:", "Integration time:", "Lag time:",
            "Z-Position (Manual):"]

LABELS = {0: "Fluorescein",
          1: "Rhodamine"}

class PipetteTutorial:
    def __init__(self, excelname, save_loc, labels=2):
        excel = os.path.join(save_loc, excelname)
        self.df = pd.read_excel(excel, sheet_name=0)
        self.metadata = self.parseMetadata(labels)
        self.data = self.filterData(self.parseData(labels))
    def parseMetadata(self, labels):
        df = self.df
        all_metadata = []
        for label in range(labels):
            metadata = {}
            def parse(name):
                df2 = df.loc[df[df.columns[0]]==name].dropna(axis=1)
                metadata_val = df2[df2.columns[1]].iloc[0]
                return metadata_val
            for item in METADATA:
                metadata[item] = parse(item)
            all_metadata.append(metadata)
        return all_metadata
    def parseData(self, labels):
        all_data = []
        for label in range(labels):
            data = {}
            df = self.df
            start_index = df.loc[df[df.columns[0]]=='<>'].index[label]
            end_index = start_index + 9
            df2 = df.iloc[start_index+1:end_index]
            index = 'A'
            for row in df2.iterrows():
                df_row = list(row[1][1:])
                if "..." not in df_row:
                    if df_row[0] < 100:
                        continue
                    data[index] = df_row
                    index = chr(ord(index) + 1)
            all_data.append(pd.DataFrame(data))
        return all_data
    def dilutionLine(self, row, label, save_loc):
        Output = []
        plt.figure(figsize=(5,5))
        plt.plot(self.data[label][row])
        plt.title(row + " " + LABELS[label])
        Output.append(LABELS[label] + " " + str(row) + ": " +  str(linregress(self.data[label][row], self.data[label].index)))
        Output.append("Logistic Regression (mean, variance)" + ": " + str(logistic.fit(list(self.data[label][row]))))
        plt.savefig(os.path.join(save_loc, row + " " + LABELS[label]+"_lineplot.png"))
        plt.close()
        return Output
    def filterData(self, df_arr):
        output_arr = []
        self.norms = []
        def findNorms(df):
            norm = {}
            for norm_index in range(len(df.columns)):
                norm[df.columns[norm_index]] = np.mean([int(df.iloc[0][norm_index]), int(df.iloc[5][norm_index])])
            return norm
        for index in range(len(df_arr)):
            df = df_arr[index]
            #Label 1, Fluorescein
            if index == 0 and not df.empty:
                df = df.drop(np.arange(6, 12))
                self.norms.append(findNorms(df))
                df.loc[0] = pd.Series(self.norms[0])
                df = df[:-1]
            #Label 2, Rhodamine
            elif index == 1 and not df.empty:
                df = df.drop(np.arange(0,6))
                df = df.reset_index(drop=True)
                self.norms.append(findNorms(df))
                df.loc[0] = pd.Series(self.norms[1])
                df = df[:-1]
            output_arr.append(df.astype('float64'))
        return output_arr


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class NNModel:
    def __init__(self):
        self.model = None

    def formatData(self, df, cat):
        data = df.drop(cat, axis=1)
        vals = df[cat]
        return data, vals
    def MLP(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(15,10,10),
            max_iter=250)
    def RandomForest(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=2,
            random_state=0)
    def fitData(self, train, label):
        self.model.fit(train, label)

    def predict(self, df):
        return self.model.predict(df)
    def predictAnalysis(self, df, correctdf):
        predictions = self.predict(df)
        print(confusion_matrix(correctdf, predictions))
        print(classification_report(correctdf, predictions))

#Other analysis
def slope_intercept(x_val, y_val):
    x=np.array(x_val)
    y=np.array(y_val)
    m=(((np.mean(x)*np.mean(y))) - np.mean(x*y)) / ((np.mean(x)*np.mean(x)) - np.mean(x*x))
    m=round(m, 2)
    b=(np.mean(y) - np.mean(x)*m)
    b = round(b, 2)
    return m, b

def target(student):
    return [student[0], student[0]/2, student[0]/4, student[0]/8, student[0]/16, student[0]]

def plot_student(student, save_loc, filename):
    x = [1, 1/2, 1/4, 1/8, 1/16, 1]
    labels = ["1-norm", "2", "3", "4", "5", "6"]
    diff = np.array(target(student)) * 0.21
    m, b = slope_intercept(x, student)
    reg_line=[(m*i)+b for i in x]
    plt.figure(figsize=(5,5))
    plt.scatter(x, student, color="r", s = 12)
    plt.plot(x, target(student))
    plt.plot(x, target(student) - diff, linewidth=.3)
    plt.title("Pipetting Accuracy Detection")
    plt.xlabel("Dilution")
    plt.ylabel("Read")
    plt.plot(x, target(student) + diff, linewidth=.3)
    for i in range(len(student)):
        plt.text(x[i] + 0.01, student[i], labels[i])
    plt.savefig(os.path.join(save_loc, filename))
    plt.clf()
    return filename

def scoring(student, row):
    diff = np.array(target(student)) * 0.21
    student_diff = abs(np.array(student) - np.array(target(student)))
    passes = list(student_diff <= diff)
    df = pd.DataFrame(passes).T
    df.columns = [1, 2, 3, 4, 5, 6]
    df = df.replace({True: 'Pass', False: 'Fail'})
    df = df.rename(index={0: row})
    return df

def runAnalysis(excelname, save_loc, model_directory):
    exp = PipetteTutorial(excelname, save_loc)
    documentation = {}
    results = {}

    #Dilution lines
    #Predict Accuracy / Problem
    fluorescein = NNModel()
    rhodamine = NNModel()
    with open('fluorescein.pkl', 'rb') as fid:
        fluorescein.model = pickle.load(fid)
    with open('rhodamine.pkl', 'rb') as fid:
        rhodamine.model = pickle.load(fid)

    prediction = {1: "Pipetting was done correctly.",
                  0: "Incorrect: Pipetting under the correct amount (not enough liquid).",
                  2: "Incorrect: Pipetting over the correct  amount (too much liquid)."}
    additional = []
    for label in range(len(exp.data)):
        index = 'A'
        for _ in exp.data[label]:
            documentation[index + " " + LABELS[label]] = exp.dilutionLine(index, label, save_loc)
            if label == 0:
                results[index + " " + LABELS[label]] = prediction[fluorescein.predict(exp.data[label][[index]].T)[0]]
            elif label == 1:
                results[index + " " + LABELS[label]] = prediction[rhodamine.predict(exp.data[label][[index]].T)[0]]
            student = list(exp.data[label][index])
            student = student + [student[0]]
            filename = str(index + " " + LABELS[label] + "additional.png")
            additional.append([scoring(student, index), plot_student(student, save_loc, filename)])
            index = chr(ord(index) + 1)

    return documentation, results, exp.metadata, additional
