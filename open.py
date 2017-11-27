import csv
import locale
from functools import cmp_to_key
import csv
import numpy as np
import plotly.plotly as py

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import seaborn
import seaborn as sns

import pandas as pd
import glob
import os
import re


from pandas import Series
from matplotlib import pyplot

import seaborn

import numpy as np
import pandas as pd
import matplotlib

from matplotlib.mlab import csv2rec
matplotlib.style.use('ggplot')       # Use ggplot style plots*
from matplotlib.cbook import get_sample_data

directoryPath='/home/allan/PycharmProjects/datamining/dados_consumo_todos/total/'

def main():
    # df = pd.read_csv('/home/allan/PycharmProjects/datamining/dados_consumo_todos/total/fee-1996-mun-consumo-total-100849.csv', encoding="ISO-8859-1", header=None)
    correlation()
    normalizar()
    graficoScatter()
    graficoLine()


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

def correlation():
    with open('/home/lucas/PycharmProjects/openCsv/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])


    anos = anos[['municipio','mwh','ano']]
    cols_to_norm = ['mwh']
    anos[cols_to_norm] = anos[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    # pivoted = anos.pivot('ano', 'municipio')
    pivoted = anos.pivot(index='municipio', columns='ano', values='mwh')
    # Compute the correlation matrix
    corr = pivoted.corr()
    ax = sns.heatmap(corr)
    # print(corr)
    # anossort = anos.sort_values(by='ano', ascending=True)
    # anos['mwh'] = anos[['mwh']].astype(float)
    pivoted = pivoted.fillna(0)
    # train_float = pivoted.select_dtypes(include=['float64'])
    # colormap = plt.cm.magma
    # plt.figure(figsize=(16, 12))
    # plt.title('Pearson correlation of continuous features', y=1.05, size=15)
        ax = sns.heatmap(pivoted)
    # sns.heatmap(train_float.corr(), linewidths=0.1, vmax=1.0, square=True,
    #             cmap=colormap, linecolor='white', annot=True)


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

def graficoLine():
    with open('/home/lucas/PycharmProjects/openCsv/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])

    nomes =[]
    for index, row in anos.iterrows():
        nomes.append(row["municipio"])
        if(index==460):
            break

    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 1, len(nomes))]
    labels = []

    plt.xlim([1990, 2015])
    # plt.ylim([0, 10])
    for rank , colunm in enumerate(nomes):
        selecionados = anos['municipio'] == colunm
        umframe = anos[selecionados]
        umframesort = umframe.sort_values(by='ano', ascending=True)
        umframesort = umframesort[['mwh', 'ano']].astype(int)
        plt.plot(umframesort['ano'],umframesort['mwh'],'k', color = colors[rank])
        labels.append(colunm)
        # ax = umframesort.plot(x='ano', y='mwh', style='.-')

    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)

    plt.show()



def graficoScatter():
    with open('/home/lucas/PycharmProjects/openCsv/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])

    for index, row in anos.iterrows():
        row["mwh"] = int(row["mwh"])

    anossort = anos.sort_values(by='ano', ascending=True)
    anostype = anossort[['mwh', 'ano']].astype(float)
    ax =anostype.plot(x='ano', y='mwh',style='k.')
    ax.set_xlim(1990,2016)

    plt.show()


def normalizar():
    with open('/home/lucas/PycharmProjects/openCsv/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])

    # for index, row in anos.iterrows():
    #     row["mwh"] = int(row["mwh"])

    anossort = anos.sort_values(by='ano', ascending=True)
    anossort['mwh'] = anossort[['mwh']].astype(float)
    #This will apply it to only the columns you desire
    cols_to_norm = ['mwh']
    anossort[cols_to_norm] = anossort[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    ax = anossort.plot(x='ano', y='mwh', style='k.')
    ax.set_xlim(1990, 2016)
    nomes = []
    for index, row in anos.iterrows():
        nomes.append(row["municipio"])
        if (index == 460):
            break

    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 1, len(nomes))]
    labels = []

    plt.xlim([1990, 2016])
    for rank, colunm in enumerate(nomes):
        selecionados = anossort['municipio'] == colunm
        umframe = anossort[selecionados]
        umframesort = umframe.sort_values(by='ano', ascending=True)
        # umframesort = umframesort[['mwh', 'ano']].astype(float)
        plt.plot(umframesort['ano'], umframesort['mwh'], 'k', color=colors[rank])
        labels.append(colunm)

    plt.show()



def graficoTes():

    with open('/home/lucas/PycharmProjects/openCsv/dados_consumo_todos/total.csv', encoding="ISO-8859-1") as fname:
        gender_degree_data = csv2rec(fname)
    anos = pd.DataFrame(gender_degree_data, columns=['municipio', 'ibge', 'latitude', 'longitude', 'mwh', 'ano'])

    for index, row in anos.iterrows():
        # if(len(str(row["mwh"]))>6):
        # cells = str(row["mwh"])
        # for rank, c in cells:
        row["mwh"] = int(row["mwh"])
        # if(rank!=6):
        #   cells[rank] = '.'
        # cells[rank]=c

    grouped = anos.groupby('municipio')
    print(grouped)
    anossort = anos.sort_values(by='ano', ascending=True)
    anostype = anossort[['mwh', 'ano']].astype(float)
    ax = anostype.plot(x='ano', y='mwh', style='k.')
    ax.set_xlim(1990, 2016)

    american = anos['municipio'] == "Agudo"
    print(anos[american])
    select = anos[american]

    nomes = []
    for index, row in anos.iterrows():
        nomes.append(row["municipio"])
        if (index == 400):
            break

    colormap = plt.cm.gist_ncar  # nipy_spectral, Set1,Paired
    colors = [colormap(i) for i in np.linspace(0, 1, len(nomes))]
    labels = []

    plt.xlim([1990, 2015])
    plt.ylim([0, 10])
    for rank, colunm in enumerate(nomes):
        selecionados = anos['municipio'] == colunm
        umframe = anos[selecionados]
        umframesort = umframe.sort_values(by='ano', ascending=True)
        umframesort = umframesort[['mwh', 'ano']].astype(int)
        plt.plot(umframesort['ano'], umframesort['mwh'], 'k', color=colors[rank])
        labels.append(colunm)
        # ax = umframesort.plot(x='ano', y='mwh', style='.-')

    plt.legend(labels, ncol=4, loc='upper center',
               bbox_to_anchor=[0.5, 1.1],
               columnspacing=1.0, labelspacing=0.0,
               handletextpad=0.0, handlelength=1.5,
               fancybox=True, shadow=True)

    plt.show()

    # anos[american]
    # select3=select.sort_values(by='ano', ascending=True)
    # select3  = select3[['mwh','ano']].astype(float)

    # print(select3)
    # # yaxis = select[['mwh']]
    # # yaxis=yaxis.values
    # select3.plot(x='ano', y ='mwh' , style='.-')



if __name__ == "__main__":
    main()