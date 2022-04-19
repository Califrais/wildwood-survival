import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from lifelines import KaplanMeierFitter

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