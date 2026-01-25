import math
from typing import Tuple, List, Dict, Any

import numpy as np


def score_bell_column(
    weights: np.ndarray,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
    # Score Bell для одного критерия; возвращает SB-баллы и параметры колокола.
    w = np.asarray(weights, dtype=float)
    xmin = float(w.min())
    xmax = float(w.max())
    Q1 = float(np.percentile(w, 25))
    Q3 = float(np.percentile(w, 75))

    # SICP = среднее по значениям внутри [Q1, Q3]
    mask_iz = (w >= Q1) & (w <= Q3)
    sicp = float(w[mask_iz].mean())

    scores = np.empty_like(w, dtype=float)

    for i, val in enumerate(w):
        # Зона 1: [min, Q1)
        if val < Q1:
            xrel = val - xmin
            denom = Q1 - xmin or 1e-9
            scores[i] = math.exp(math.log(s_mean) * (xrel / denom) ** shape)

        # Зона 2: [Q1, SICP)
        elif val < sicp:
            xrel = val - Q1
            denom = sicp - Q1 or 1e-9
            scores[i] = (s_max + 1) - math.exp(
                math.log(s_max + 1 - s_mean) * (1 - xrel / denom) ** shape
            )

        # Зона 3: [SICP, Q3]
        elif val <= Q3:
            xrel = val - sicp
            denom = Q3 - sicp or 1e-9
            scores[i] = (s_max + 1) - math.exp(
                math.log(s_max + 1 - s_mean) * (xrel / denom) ** shape
            )

        # Зона 4: (Q3, max]
        else:
            xrel = val - Q3
            denom = xmax - Q3 or 1e-9
            scores[i] = math.exp(math.log(s_mean) * (1 - xrel / denom) ** shape)

    return scores, (xmin, Q1, sicp, Q3, xmax)


def expand_by_influence(
    W: np.ndarray,
    influence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Vote-casting: раздуваем матрицу W, повторяя строки пропорционально влиянию.
    W = np.asarray(W, float)
    influence = np.asarray(influence, float)

    if influence.shape[0] != W.shape[0]:
        raise ValueError("influence length must match number of rows in W")
    if np.any(influence <= 0):
        raise ValueError("influence values must be positive")

    alpha_min = float(influence.min())
    # число голосов = влияние / минимальное влияние
    votes = np.rint(influence / alpha_min).astype(int)

    owners = np.concatenate([np.full(v, k, dtype=int) for k, v in enumerate(votes)])
    W_exp = np.repeat(W, votes, axis=0)
    return W_exp, owners, votes


def hives(
    W: np.ndarray,
    influence: np.ndarray | None = None,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
) -> Dict[str, Any]:
    # Основной расчёт HIVES для матрицы W (влияния экспертов и Score Bell).
    W = np.asarray(W, float)
    m, n = W.shape

    # Если влияние не задано — все эксперты равны
    if influence is None:
        W_exp = W
        owners = np.arange(m, dtype=int)
    else:
        W_exp, owners, _votes = expand_by_influence(W, influence)

    # Score Bell на "раздутой" матрице
    S_exp = np.zeros_like(W_exp)
    params = []
    for j in range(n):
        scores_j, params_j = score_bell_column(
            W_exp[:, j], s_mean=s_mean, s_max=s_max, shape=shape
        )
        S_exp[:, j] = scores_j
        params.append(params_j)

    # Складываем баллы обратно по исходным экспертам
    S = np.zeros((m, n))
    for i in range(m):
        S[i, :] = S_exp[owners == i, :].sum(axis=0)

    # Веса экспертов по критериям (λ_ij)
    col_sums = S.sum(axis=0, keepdims=True)
    lambdas = S / col_sums * 100

    # Веса критериев γ_j
    gamma = (lambdas * W).sum(axis=0) / 100

    # Масштабируем до суммы 100
    beta = 100.0 / float(gamma.sum())
    gamma_scaled = gamma * beta

    return dict(
        gamma=gamma,
        gamma_scaled=gamma_scaled,
        lambdas=lambdas,
        scores=S,
        params=params,
    )


def hives_rank(
    A: np.ndarray,
    W: np.ndarray,
    influence: np.ndarray | None = None,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
) -> Dict[str, Any]:
    # Обёртка над hives(): считает итоговые оценки и ранжирование альтернатив.
    A = np.asarray(A, float)
    res = hives(W, influence=influence, s_mean=s_mean, s_max=s_max, shape=shape)
    gamma_scaled = res["gamma_scaled"]

    # Вклад по каждому критерию
    alt_details = A * gamma_scaled  # по статье просто умножают
    alt_scores = alt_details.sum(axis=1)
    ranking = np.argsort(-alt_scores)  # по убыванию

    res.update(
        dict(
            alt_scores=alt_scores,
            alt_details=alt_details,
            ranking=ranking,
        )
    )
    return res


