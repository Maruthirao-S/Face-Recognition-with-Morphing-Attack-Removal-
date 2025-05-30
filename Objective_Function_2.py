import numpy as np
from  Global_vars import Global_vars
from Model_DTCN import Model_DTCN

def Objective_Function_2(soln):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    for k in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = soln[k,:].astype('int')
        else:
            sol = soln.astype('int')
        learnper = round(Global_vars.Features.shape[0] * 0.75)
        train_data = Global_vars.Features[:learnper, :]
        train_target = Global_vars.Target[:learnper, :]
        test_data = Global_vars.Features[learnper:, :]
        test_target = Global_vars.Target[learnper:, :]

        Eval = Model_DTCN(train_data, train_target, test_data, test_target, sol) # Deep Temporal Convolution Network
        fitn[k] = (1 / (Eval[0][4] + Eval[0][7])) + Eval[0][9]+Eval[0][8] + Eval[0][12]
    return fitn

