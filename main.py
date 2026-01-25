from __future__ import annotations

import argparse

import numpy as np

from json_input import load_decision_problem_from_json
from hives_method import hives_rank


def run_from_json(json_path: str) -> None:
    # Запуск расчёта по JSON: читаем данные, считаем HIVES, печатаем результаты.
    problem = load_decision_problem_from_json(json_path)

    # Матрица A (альтернатива × критерий), агрегированная по всем ЛПР
    A = problem.aggregated_performance()

    # Матрица W (эксперт × критерий)
    W = problem.weights_matrix()

    res = hives_rank(A=A, W=W)

    print("\nСырые веса критериев γ:")
    print(np.round(res["gamma"], 2))

    print("\nНормированные веса критериев γ' (в сумме = 100):")
    print(np.round(res["gamma_scaled"], 2))

    print("\nИтоговые баллы альтернатив:")
    for idx, score in enumerate(np.round(res["alt_scores"], 2)):
        print(f"  {problem.alternatives[idx]}: {score}")

    print("\nРанжирование альтернатив (1-based):")
    ranking_1_based = res["ranking"] + 1
    print(ranking_1_based)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Групповой выбор альтернатив по методу HIVES (входные данные в JSON)."
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Путь к JSON-файлу с входными данными.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_from_json(args.json_path)

