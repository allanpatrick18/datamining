import csv
import locale
from functools import cmp_to_key
import csv
import numpy as np
import plotly.plotly as py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import seaborn as sns

import pandas as pd
import glob
import os
import re

import sns as sns
from pandas import Series
from matplotlib import pyplot

import seaborn

import numpy as np
import pandas as pd
import matplotlib


matplotlib.style.use('ggplot')       # Use ggplot style plots*



directoryPath='/home/allan/PycharmProjects/datamining/dados_consumo_todos/total/'

def main():

    # demo()
    df = pd.read_csv('/home/allan/PycharmProjects/datamining/dados_consumo_todos/total.csv', encoding="ISO-8859-1", header=None)
    # df = pd.DataFrame(df2,  columns=list('ABCD'))
    # print(df)


    df1 = df.ix[:,5:6]
    df.plot(x='MWh', y='ano')
    df1 = df1.values
    df = df.cumsum()
    plt.figure();
    df.plot();

    # list(df1)
    # print(df1)
    # df1 = df.ix[:,6]
    # df1 = df1.values
    # list(df1)
    # series = df
    # series.hist()
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

def demo():
    N = 100

    num_runs = 3
    out = []
    for k in range(num_runs):
        data = np.random.rand(N, 3) + np.sin(np.arange(N) / 5)[:, np.newaxis]
        data = np.hstack([np.arange(N)[:, np.newaxis], data])
        data = np.hstack([np.zeros(N)[:, np.newaxis] + k, data])
        out.append(data)

    data = np.vstack(out)

    df = pd.DataFrame(data, columns=['sub', 't', 'x', 'y', 'z'])
    dfm = pd.melt(df, id_vars=['t', 'sub'], value_vars=['x', 'y', 'z'])
    dfm
    seaborn.tsplot(time='t',
               value='value',
               condition='variable',
               data=dfm,
               err_style="boot_traces",
               unit='sub',
               n_boot=50)


if __name__ == "__main__":
    main()