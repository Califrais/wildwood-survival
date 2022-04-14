from sksurv.tree import SurvivalTree
from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
def tree_based_estimators_(estimators_list, max_depth = 3,
                           min_samples_leaf = 1, n_estimators = 100):
    """
    List tree-based estimators of competing models

    estimators_list : `list`
        List of estimator's names, it requires the name to be in the
        list of defined names ["SurvivalTree", "RSF", "GBSA"]

    max_depth : `int`, default: 3
        Maximum depth of the individual trees.M

    min_samples_leaf : `int`, default: 1
        The minimum number of samples required to be at a leaf node.M

    n_estimators : `int`, default: 100
        The number of regression trees to create.

    Returns
    -------
    estimators : `dict`
        Dictionary of estimators.

    """
    max_depth_ = max_depth
    min_samples_leaf_ = min_samples_leaf
    n_estimators_ = n_estimators
    estimators = {}
    for estimator in estimators_list:
        if estimator == "SurvivalTree":
            estimators[estimator] = SurvivalTree(max_depth = max_depth_,
                                        min_samples_leaf = min_samples_leaf_)
        elif estimator == "RSF":
            estimators[estimator] = RandomSurvivalForest(
                                                n_estimators = n_estimators_)
        elif estimator == "GBSA":
            estimators[estimator] = GradientBoostingSurvivalAnalysis(
                                                n_estimators = n_estimators_)
        else:
            raise Exception('Can not find {} estimator in the list of supported'
                            ' names'.format(estimator))

    return estimators