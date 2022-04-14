from data_loader import real_data, simulated_data
import numpy as np

def load_data(data_list):
    """
    List tree-based estimators of competing models

    data_list : `list`
        List of estimator's names, it requires the name to be in the
        list of defined names ["GBSG", "PBCSeq", "simple_simulated_data",
        "linear_simulated_data", "non_linear_simulated_data"]

    Returns
    -------
    datas : `dict`
        Dictionary of dataset.
    """

    datas = {}

    for data in data_list:
        if data == "GBSG":
            # The German Breast Cancer Study Group 2 dataset
            X, y = real_data.GBSG()

        elif data == "PBCSeq":
            # The PBC_Seq dataset
            X, y = real_data.PBC_Seq()

        elif data == "simple_simulated_data":
            X, y = simulated_data.simple_simulated_data()


        elif data == "linear_simulated_data":
            X, y = simulated_data.linear_simulated_data()

        elif data == "non_linear_simulated_data":
            X, y = simulated_data.non_linear_simulated_data()

        else:
            raise Exception('Can not find {} data in the list of supported names'
                                .format(data))
        datas[data] = (X, y)

    return datas