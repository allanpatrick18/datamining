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

from matplotlib.mlab import csv2rec
matplotlib.style.use('ggplot')       # Use ggplot style plots*
from matplotlib.cbook import get_sample_data

color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
directoryPath='/home/allan/PycharmProjects/datamining/dados_consumo_todos/total/'

def main():


    # df = pd.read_csv('/home/allan/PycharmProjects/datamining/dados_consumo_todos/total/fee-1996-mun-consumo-total-100849.csv', encoding="ISO-8859-1", header=None)
    caso()
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
                prev =result
            else:
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

def caso():
    with open('/home/allan/PycharmProjects/datamining/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])

    for index, row in anos.iterrows():
        if(len(str(row["mwh"]))>6):
           cells = str(row["mwh"])
           for rank, c in cells:
                if(rank!=6):
                  cells[rank] = '.'
                cells[rank]=c





    anossort = anos.sort_values(by='ano', ascending=True)
    anostype = anossort[['mwh', 'ano']].astype(int)
    anostype.plot(x='ano', y='mwh',style='k.')



    american = anos['municipio'] == "Agudo"
    print(anos[american])
    select = anos[american]

    nomes =[]
    for index, row in anos.iterrows():
        nomes.append(row["municipio"])
        if(index==10):
            break

    for rank , colunm in enumerate(nomes):
        selecionados = anos['municipio'] == colunm
        umframe = anos[selecionados]
        umframesort = umframe.sort_values(by='ano', ascending=True)
        umframesort = umframesort[['mwh', 'ano']].astype(float)
        umframesort.plot(x='ano', y='mwh', style='.-')

    plt.show()
    anos[american]
    select3=select.sort_values(by='ano', ascending=True)
    select3  = select3[['mwh','ano']].astype(float)

    print(select3)
    # yaxis = select[['mwh']]
    # yaxis=yaxis.values
    select3.plot(x='ano', y ='mwh' , style='.-')




if __name__ == "__main__":
    main()