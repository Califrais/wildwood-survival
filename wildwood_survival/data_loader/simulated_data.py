import numpy as np

def simple_simulated_data():
    """
    The survival time T is simulated in simple way.

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix.

    y : structured array, shape = (n_samples, )
        A structured array containing the simulated censoring indicator as first
         field, and the simulated censored times of the event of interest
         as second field.
    """
    # TODO: Add arguments into function
    n_features = 2
    n_groups = 3
    n_samples = 100
    p = np.array([.3, .5, .2])
    n_samples_group = (p * n_samples).astype(int)
    G = []
    for g in range(n_groups):
        G += [g] * n_samples_group[g]
    G = np.array(G)

    # create features X
    X = np.zeros((n_samples, n_features))
    X[:, 0][G == 0] = np.random.normal(5, 1, n_samples_group[0])
    X[:, 1][G == 0] = np.random.normal(-2, 1, n_samples_group[0])
    X[:, 0][(G == 1) | (G == 2)] = np.random.normal(-1, 1, n_samples_group[1] + n_samples_group[2])
    X[:, 1][G == 1] = np.random.normal(3, 1, n_samples_group[1])
    X[:, 1][G == 2] = np.random.normal(-2, 1, n_samples_group[2])

    # create survival-time and censoring indicator
    Y = np.zeros(n_samples)
    delta = np.ones(n_samples) == 1
    Y[G == 0] = np.random.normal(30, 2, n_samples_group[0])
    Y[G == 1] = np.random.normal(20, 2, n_samples_group[1])
    Y[G == 2] = np.random.normal(10, 2, n_samples_group[2])

    y = np.zeros(n_samples, dtype={'names':('indicator', 'time'), 'formats':('?', 'f8')})
    y['indicator'] = delta
    y['time'] = Y

    return X, y

def linear_simulated_data():
    """
    The survival time T is simulated according to an exponential Cox model
    in which the regression term in form linear function.

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix.

    y : structured array, shape = (n_samples, )
        A structured array containing the simulated censoring indicator as first
         field, and the simulated censored times of the event of interest
         as second field.
    """
    # TODO: Add arguments into function
    n_samples = 1000
    n_features = 50
    baseline = 10
    censoring_factor = 5
    coefs = np.array([.1, .2, .3])
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    h = X[:, :len(coefs)].dot(coefs)
    T_star = np.exp(baseline * h)
    m = T_star.mean()
    # Simulation of the censoring
    C = np.random.exponential(scale=censoring_factor * m, size=n_samples)
    # Observed censored time
    T = np.minimum(T_star, C)
    # Censoring indicator: 1 if it is a time of failure, 0 if censoring
    delta = (T_star <= C).astype(np.ushort)
    y = np.array(list(zip(delta, T)), dtype=[('indicator', '?'), ('time', 'f8')])

    return X, y

def non_linear_simulated_data():
    """
    The survival time T is simulated according to an exponential Cox model
    in which the regression term in form exponential function.

    Returns
    -------
    X : `numpy.ndarray`, shape=(n_samples, n_features)
        The simulated features matrix.

    y : structured array, shape = (n_samples, )
        A structured array containing the simulated censoring indicator as first
         field, and the simulated censored times of the event of interest
         as second field.
    """
    # TODO: Add arguments into function
    n_samples = 1000
    n_features = 50
    baseline = 10
    censoring_factor = 5
    sigma = .8
    X = np.random.uniform(-1, 1, (n_samples, n_features))
    h = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-X[:, :1]**2 / (2 * sigma**2))
    T_star = np.exp(baseline * h).flatten()
    m = T_star.mean()
    # Simulation of the censoring
    C = np.random.exponential(scale=censoring_factor * m, size=n_samples)
    # Observed censored time
    T = np.minimum(T_star, C)
    # Censoring indicator: 1 if it is a time of failure, 0 if censoring
    delta = (T_star <= C).astype(np.ushort)
    y = np.array(list(zip(delta, T)), dtype=[('indicator', '?'), ('time', 'f8')])

    return X, y