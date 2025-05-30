import numpy as np
import time


def CO(X, fitness, LB, UB, Max_iterations):
    SearchAgents, dimension = X.shape
    lowerbound, upperbound = LB[0, :], UB[0, :]

    # Evaluate initial population
    fit = [fitness(L) for L in X]

    # Initialize best solution
    fbest = float('inf')
    Xbest = None

    best_so_far = np.zeros(Max_iterations)
    average = np.zeros(Max_iterations)

    ct = time.time()
    for t in range(Max_iterations):
        # Update the best candidate solution
        best, location = min(zip(fit, range(SearchAgents)))

        if t == 0 or best < fbest:
            fbest = best
            Xbest = X[location, :]

        # Phase 1: Hunting and attacking strategy on iguana (Exploration Phase)
        for i in range(SearchAgents // 2):
            iguana = Xbest
            I = round(1 + np.random.rand())
            X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (4)
            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        # Phase 2: The process of escaping from predators (Exploitation Phase)
        for i in range(SearchAgents // 2, SearchAgents):
            iguana = lowerbound + np.random.rand() * (upperbound - lowerbound)  # Eq. (5)
            L = iguana
            F_HL = fitness(L)
            I = round(1 + np.random.rand())

            if fit[i] > F_HL:
                X_P1 = X[i, :] + np.random.rand() * (iguana - I * X[i, :])  # Eq. (6)
            else:
                X_P1 = X[i, :] + np.random.rand() * (X[i, :] - iguana)  # Eq. (6)

            X_P1 = np.maximum(X_P1, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        # Phase 3: The process of movement and random position selection
        t_divided_by_max_iterations = (t + 1) / Max_iterations
        LO_LOCAL = lowerbound / t_divided_by_max_iterations  # Eq. (9)
        HI_LOCAL = upperbound / t_divided_by_max_iterations  # Eq. (10)

        for i in range(SearchAgents):
            X_P2 = X[i, :] + (1 - 2 * np.random.rand()) * (
                        LO_LOCAL + np.random.rand() * (HI_LOCAL - LO_LOCAL))  # Eq. (8)
            X_P2 = np.maximum(X_P2, LO_LOCAL)
            X_P2 = np.minimum(X_P2, HI_LOCAL)

            L = X_P2
            F_P2 = fitness(L)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        best_so_far[t] = fbest
        average[t] = np.mean(fit)

    Best_score = fbest
    Best_pos = Xbest
    COA_curve = best_so_far
    ct = time.time() - ct
    return Best_score, COA_curve, Best_pos, ct
