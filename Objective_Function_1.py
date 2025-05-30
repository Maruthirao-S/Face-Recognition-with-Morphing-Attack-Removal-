import numpy as np
from Global_vars import Global_vars
from AGANI import AGANI


def Objective_Function_1(soln):
    if soln.ndim == 2:
        dim = soln.shape[1]
        v = soln.shape[0]
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0];
        v = 1
        fitn = np.zeros((1, 1))

    for k in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = soln[k, :]
        else:
            sol = soln

        impainted = AGANI(Global_vars.Real_Images, Global_vars.Fake_Images,
                          sol)  # Occlusion removal using Adaptive Auxiliary GAN Inversion
        Total = len(Global_vars.Fake_Images)  # No of occluded images
        solved = len(impainted)  # No of occlusion removed images
        Occlusion_Solved_Rate = solved / Total
        fitn[k] = 1 / Occlusion_Solved_Rate
    return fitn
