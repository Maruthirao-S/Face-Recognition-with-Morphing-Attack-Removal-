import numpy as np
import time


def CSA(chameleonPositions, fobj, lb, ub, iteMax):
    searchAgents, dim = chameleonPositions.shape
    if len(ub) == 1:
        ub = np.ones(dim) * ub
        lb = np.ones(dim) * lb

    cg_curve = np.zeros(iteMax)

    fit = np.zeros(searchAgents)
    for i in range(searchAgents):
        fit[i] = fobj(chameleonPositions[i, :])

    fitness = fit
    fmin0, index = min(fit), np.argmin(fit)
    chameleonBestPosition = chameleonPositions.copy()
    gPosition = chameleonPositions[index, :]
    v = 0.1 * chameleonBestPosition
    v0 = 0.0 * v

    rho = 1.0
    p1, p2, c1, c2, gamma, alpha, beta = 2.0, 2.0, 2.0, 1.80, 2.0, 4.0, 3.0
    ct = time.time()
    for t in range(iteMax):
        a = 2590 * (1 - np.exp(-np.log(t + 1)))
        omega = (1 - (t / iteMax)) ** (rho * np.sqrt(t / iteMax))
        p1 = 2 * np.exp(-2 * (t / iteMax) ** 2)
        p2 = 2 / (1 + np.exp((-t + iteMax / 2) / 100))
        mu = gamma * np.exp(-(alpha * t / iteMax) ** beta)
        ch = np.random.randint(searchAgents, size=searchAgents)

        for i in range(searchAgents):
            if np.random.rand() >= 0.1:
                chameleonPositions[i, :] = chameleonPositions[i, :] + p1 * (
                        chameleonBestPosition[ch[i], :] - chameleonPositions[i, :]) * np.random.rand() + p2 * (
                                                   gPosition - chameleonPositions[i, :]) * np.random.rand()
            else:
                for j in range(dim):
                    chameleonPositions[i, j] = gPosition[j] + mu * (
                            (ub[j] - lb[j]) * np.random.rand() + lb[j]) * np.sign(np.random.rand() - 0.50)

        for i in range(searchAgents):
            v[i, :] = omega * v[i, :] + p1 * (
                    chameleonBestPosition[i, :] - chameleonPositions[i, :]) * np.random.rand() + p2 * (
                              gPosition - chameleonPositions[i, :]) * np.random.rand()
            chameleonPositions[i, :] = chameleonPositions[i, :] + (v[i, :] ** 2 - v0[i, :] ** 2) / (2 * a)

        v0 = v

        for i in range(searchAgents):
            chameleonPositions[i, :] = np.where(chameleonPositions[i, :] < lb, lb, chameleonPositions[i, :])
            chameleonPositions[i, :] = np.where(chameleonPositions[i, :] > ub, ub, chameleonPositions[i, :])

        for i in range(searchAgents):
            ub_ = np.sign(chameleonPositions[i, :] - ub) > 0
            lb_ = np.sign(chameleonPositions[i, :] - lb) < 0
            chameleonPositions[i, :] = (chameleonPositions[i, :] * np.logical_not(
                np.logical_xor(lb_, ub_))) + ub * ub_ + lb * lb_

            fit[i] = fobj(chameleonPositions[i, :])

            if fit[i] < fitness[i]:
                chameleonBestPosition[i, :] = chameleonPositions[i, :]
                fitness[i] = fit[i]

        fmin, index = min(fitness), np.argmin(fitness)
        if fmin < fmin0:
            gPosition = chameleonBestPosition[index, :]
            fmin0 = fmin

        cg_curve[t] = fmin0

    ngPosition = np.where(fitness == min(fitness))[0]
    g_best = chameleonBestPosition[ngPosition[0], :]
    fmin0 = fobj(g_best)
    ct = time.time() - ct
    return fmin0, cg_curve, gPosition, ct
