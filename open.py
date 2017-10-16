import csv
import locale
from functools import cmp_to_key
import csv
import numpy as np
import plotly.plotly as py

import pandas as pd
import glob
import os
import re
from pandas import Series
from matplotlib import pyplot

directoryPath='/home/lucas/Downloads/dados_consumo_todos/total/'

def main():

    df = pd.read_csv('/home/lucas/Downloads/dados_consumo_todos/total.csv',encoding="ISO-8859-1", header=None)


    df1 = df.ix[:,6]
    df1 = df1.values
    list(df1)
    print(df1)
    df1 = df.ix[:,6]
    df1 = df1.values
    list(df1)
    series = df
    series.hist()
    i =0
    os.chdir(directoryPath)
    files=glob.glob("*.csv")
    # sorted(files, key=cmp_to_key(locale.strcoll))
    # for file in files:
    #     print(file)
    for file in files:
            x = pd.read_csv(file,  encoding="ISO-8859-1", header=None)
            x.loc[:, 'ano'] = getYear(file)
            if(i!=0):
                result = x.append(prev, ignore_index=True)

            prev = x
            i += 1

    result.drop(result.index[0])
    print(result)
    result.to_csv('total', sep=',', encoding='ISO-8859-1')


def getYear(file):
    teste = re.findall(r'\d+',file)
    return teste[0]


if __name__ == "__main__":
    main()