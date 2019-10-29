import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

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
    def dilutionLine(self, row, save_loc):
        Output = []
        plt.plot(self.data[row])
        plt.title(row)
        Output.append(str(row) + ": " +  str(linregress(self.data['A'], self.data.index)))
        plt.savefig(os.path.join(save_loc, row+"_lineplot.png"))
        return Output
