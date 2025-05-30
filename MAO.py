import numpy as np
import time


def MAO(axolotl, funcc, Lower, Upper, evaluations):
    global e
    e = 0
    global cg_curve_global
    np, D = axolotl.shape[0], axolotl.shape[1]
    Lb = Lower[0, :]
    Ub = Upper[0, :]
    dp = 0.5 # Damage probability
    rp = 0.1 # Regeneration probability
    k = 3 # Tournament size
    lmbda = 0.5 # lambda value

    cg_curve_global = np.zeros(evaluations)
    cg_curve = np.zeros(evaluations)
    np.set_printoptions(precision=15)
    print("Computing ... it may take a few minutes.")

    # CrossOver Probability
    cop = 0.5

    # Initialization
    O = np.full(np, np.inf ** 10)

    fmin, best_axolotl, axolotl, O = get_best(funcc, axolotl, axolotl, O, np.arange(np))

    # Divide the population into males and females
    male = np.arange(0, np, 2)
    female = np.arange(1, np, 2)
    M = len(male)
    F = len(female)
    ct = time.time()
    # Starting evaluations
    while e < evaluations:
        # Transition procedure
        # Phase 1: Transition from larvae to adult state

        # Select the best male and female axolotls
        m = axolotl[male, :]
        K = np.argmin(O[male])
        mbest = m[K, :]

        f = axolotl[female, :]
        K = np.argmin(O[female])
        fbest = f[K, :]

        # For each male axolotl
        for j in range(M):
            pmj = O[male[j]] / np.sum(O[male])
            if pmj < np.random.rand():
                m[j, :] += (mbest - m[j, :]) * lmbda
            else:
                Lb2 = Lb - m[j, :]
                Ub2 = Ub - m[j, :]
                ran_dom = Lb2 + (Ub2 - Lb2) * np.random.rand(D)
                m[j, :] += ran_dom

        fmin, best_axolotl, axolotl, O = get_best(funcc, axolotl, m, O, male)

        # For each female axolotl
        for j in range(F):
            pfj = O[female[j]] / np.sum(O[female])
            if pfj < np.random.rand():
                f[j, :] += (fbest - f[j, :]) * lmbda
            else:
                Lb2 = Lb - f[j, :]
                Ub2 = Ub - f[j, :]
                ran_dom = Lb2 + (Ub2 - Lb2) * np.random.rand(D)
                f[j, :] += ran_dom

        fmin, best_axolotl, axolotl, O = get_best(funcc, axolotl, f, O, female)

        # Accidents procedure
        # Phase 2: Injury and restoration
        list_damage = []
        axolotl_damage = np.copy(axolotl)

        for i in range(np):
            if np.random.rand() < dp:
                for j in range(D):
                    if np.random.rand() < rp:
                        Lb2 = Lb - axolotl_damage[i, :]
                        Ub2 = Ub - axolotl_damage[i, :]
                        ran_dom = Lb2 + (Ub2 - Lb2) * np.random.rand(D)
                        axolotl_damage[i, :] += ran_dom
                        list_damage.append(i)

        list_damage = np.unique(list_damage)
        axolotl_p2 = axolotl_damage[list_damage, :]
        fmin, best_axolotl, axolotl, O = get_best(funcc, axolotl, axolotl_p2, O, list_damage)

        # NewLife procedure
        # Phase 3: Reproduction and Assortment
        f = axolotl[female, :]
        fitness_female = O[female]
        m = axolotl[male, :]
        fitness_male = O[male]

        for j in range(F):
            fj = f[j, :]
            mj, id_win = tournament_selection(k, m, fitness_male)
            egg1, egg2 = uniform_crossover(fj, mj, cop)
            e_temp = e + 1
            oegg1 = fobj(funcc, egg1)
            oegg2 = fobj(funcc, egg2)
            omj = fitness_male[id_win]
            ofj = fitness_female[j]
            ranking = np.vstack((egg1, egg2, axolotl[male[id_win]], axolotl[female[j]]))
            best, id_best = np.sort([oegg1, oegg2, omj, ofj])
            axolotl[female[j]] = ranking[id_best[0]]
            O[female[j]] = best[0]
            axolotl[male[id_win]] = ranking[id_best[1]]
            O[male[id_win]] = best[1]

            fmin, K = np.min(O), np.argmin(O)
            best = axolotl[K]
            steps = e - e_temp + 1

            if e < evaluations:
                cg_curve_global[e_temp:e] = np.repeat(fmin, steps)

        e += 1

    equal_zero = np.where(cg_curve_global == 0)
    cg_curve_global[equal_zero] = np.repeat(fmin, len(equal_zero))
    cg_curve = cg_curve_global
    ct = time.time() - ct
    return fmin, cg_curve, best, ct


def tournament_selection(k, axolotl, fitness):
    n = axolotl.shape[0]
    r = np.random.randint(0, n, k)
    select_fit = fitness[r]
    id_min = np.argmin(select_fit)
    id_win = r[id_min]
    winner = axolotl[id_win]
    return winner, id_win

def uniform_crossover(f, m, cop):
    n = len(f)
    egg1 = np.zeros(n)
    egg2 = np.zeros(n)
    for i in range(n):
        if np.random.rand() < cop:
            egg1[i] = f[i]
            egg2[i] = m[i]
        else:
            egg1[i] = m[i]
            egg2[i] = f[i]
    return egg1, egg2

def get_best(function1, axolotl, newaxolotl, fitness, ids):
    axolotl = axolotl
    fmin, K = np.min(fitness), np.argmin(fitness)
    best = axolotl[K]

    for j in range(newaxolotl.shape[0]):
        fnew = fobj(function1, newaxolotl[j])
        global e
        global cg_curve_global
        if fnew <= fitness[ids[j]]:
            fitness[ids[j]] = fnew
            axolotl[ids[j]] = newaxolotl[j]

        fmin, K = np.min(fitness), np.argmin(fitness)
        best = axolotl[K]
        cg_curve_global[e] = fmin

    return fmin, best, axolotl, fitness

def fobj(function1, u):
    global e
    z = function1(u)
    e = e + 1
    return z
