import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from lifelines import KaplanMeierFitter
import pandas as pd

def data_stats(datas, data_name):
    """
    Data analysis

    datas : `dict`
        dictionary of data

    data_name: `str`
        name of analyzed data
    """
    (X, y) = datas[data_name]
    if not isinstance(X, pd.DataFrame):
        feats = ['f' + str(i) for i in range(X.shape[1])]
        data_df = pd.DataFrame(data = X, columns = feats)
    else:
        feats = X.columns
        data_df = X
    data_df['delta'] = y['indicator'].astype(np.ushort)
    data_df['time'] = y['time']

    print('\033[1m' + "First five observations \n" + '\033[0m', data_df.head(), "\n")
    print('\033[1m' + "Shape of the data " + '\033[0m', data_df.shape, "\n")
    print('\033[1m' + "Level of censoring is " + '\033[0m', "{:.1f} %".format(100 * (1 - (data_df['delta'].sum()
                                                                                          / len(data_df['delta'])))), "\n")
    print('\033[1m' + "Data type \n" + '\033[0m', data_df.dtypes, "\n")

    print('\033[1m' + "Histogram of randomly selected features" + '\033[0m')
    if len(feats) > 3:
        sel_feats = np.sort(np.random.choice(feats, 3, replace=False))
    else:
        sel_feats = feats
    feat_visualization(data_df, sel_feats)

    print('\033[1m' + "Survival curves" + '\033[0m')
    plot_survival_curves(data_df['delta'], data_df['time'], data_name)

fontsize = 16
def plot_survival_curves(delta, T, title):
    """
    Plot survival curves of given data

    delta : np.array, shape = (n_samples, )
        Censoring indicator.

    T: np.array, shape = (n_samples, )
        Censored times of the event of interest.

    title: `str`
        Title of figure
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,3))

    # histogram plot of survival time
    bins = np.linspace(0, T.max(), 40)
    kwargs = dict(bins=bins, alpha=0.6, rwidth=0.9)
    ax[0].hist(T, **kwargs, color='r')
    ax[0].set_xlabel("T", size=fontsize)
    ax[0].set_ylabel("Count", size=fontsize)
    ax[0].set_title("Frequency histogram of T", size=fontsize)

    # Kaplan Meier estimation of survival curves
    kmf = KaplanMeierFitter()
    kmf.fit(T, delta).plot(c='r', ax=ax[1])
    ax[1].set_xlabel('Time $t$', size=fontsize)
    ax[1].set_ylabel(r'$P[S > t]$', size=fontsize)
    ax[1].set_title("Survival curves", size=fontsize)

    fig.suptitle(title, y = 1.05, size=fontsize + 1)
    plt.show()

def feat_visualization(data, sel_feats):
    """
    Visualize the data of given feature

    data : pd.DataFrame, shape = (n_samples, n_feattures)
        The given dataframe

    sel_feats: `str`
        The name of feature selected for visualization
    """
    nb_feats = len(sel_feats)
    fig, ax = plt.subplots(1, len(sel_feats), figsize=(5 * nb_feats, 3))

    # histogram plot of input features
    for i in range(nb_feats):
        data_feat = data[sel_feats[i]]
        if len(np.unique(data_feat)) < 10:
            #Categorical feature
            data_feat.value_counts().plot(kind='bar', ax=ax[i])
        else:
            bins = np.linspace(data_feat.min(), data_feat.max(), 40)
            kwargs = dict(bins=bins, alpha=0.6, rwidth=0.9)
            data_feat.plot(kind='hist', **kwargs, ax=ax[i])
        ax[i].set_title(sel_feats[i], size=fontsize)

    plt.show()