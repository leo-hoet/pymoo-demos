from typing import List

import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from plotly.graph_objs import Scatter

from pymoo.algorithms.soo.nonconvex import ga
from pymoo.core.algorithm import Algorithm
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.problems.single.knapsack import Knapsack

ITEMS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
         58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
         86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

WEIGHTS = np.array([78, 1, 13, 15, 72, 60, 82, 53, 54, 29, 28, 41, 3, 98, 2, 70, 15,
                    74, 9, 18, 49, 37, 97, 99, 15, 39, 76, 70, 86, 79, 16, 70, 46, 64,
                    27, 61, 51, 87, 50, 30, 86, 50, 5, 29, 11, 22, 63, 51, 78, 5, 44,
                    38, 28, 10, 46, 10, 56, 25, 13, 80, 7, 73, 67, 83, 59, 22, 57, 10,
                    33, 26, 16, 2, 53, 4, 44, 47, 45, 86, 2, 83, 1, 44, 27, 55, 43,
                    89, 85, 70, 5, 9, 94, 97, 57, 28, 79, 3, 32, 53, 6, 12])
PROFITS = np.array([54, 20, 94, 77, 72, 10, 78, 86, 99, 78, 7, 43, 27, 74, 64, 90, 4,
                    24, 72, 21, 14, 49, 65, 36, 19, 3, 75, 91, 34, 24, 24, 13, 20, 79,
                    50, 65, 76, 52, 18, 58, 37, 83, 55, 68, 66, 48, 15, 46, 14, 96, 2,
                    49, 35, 47, 63, 69, 12, 52, 68, 45, 57, 94, 47, 88, 46, 5, 15, 89,
                    66, 41, 75, 82, 25, 26, 82, 55, 10, 39, 79, 52, 65, 26, 51, 44, 2,
                    29, 85, 93, 44, 4, 66, 52, 55, 67, 66, 65, 76, 69, 35, 40])
N_ITEMS = len(ITEMS)
MAXIMUM_CAP = np.average(WEIGHTS)


class BestCandidateCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["feasibility"] = []
        self.data["worst"] = []

    def notify(self, algorithm):
        fitnesses: List[float] = algorithm.pop.get("F")
        constraints: List[List[float]] = algorithm.pop.get("G")
        fitnesses_constraints = zip(fitnesses, constraints)

        best_fitness, constraint = min(fitnesses_constraints, key=lambda x: x[0])

        is_feasible = all([i <= 0 for i in constraint])

        self.data["best"].append(algorithm.pop.get("F").min())
        self.data["worst"].append(algorithm.pop.get("F").max())
        self.data["feasibility"].append(is_feasible)


def combine_and_save_scatter(traces: List[Scatter], output_file: str = 'scatterplot.html', optimum_value: float = None):
    if optimum_value:
        x = traces[0].x
        y = [optimum_value] * len(x)
        traces.append(go.Scatter(x=x, y=y, mode='lines', name=f'Optimum'))

    fig = go.Figure(data=traces)
    fig.update_layout(title='Scatterplot', xaxis_title='X', yaxis_title='Y')
    pyo.plot(fig, filename=output_file)


def run_and_return_scatter(problem: Problem, algol: Algorithm, selection=None, crossover=None, n_gen=50,
                           pop_dispersion=True) -> List[Scatter]:
    res = minimize(
        problem=problem,
        algorithm=algol,
        termination=('n_gen', n_gen),
        verbose=False,
        callback=BestCandidateCallback(),
        selection=selection,
        crossover=crossover
    )
    y_values: List[float] = res.algorithm.callback.data['best']
    y_worst: List[float] = res.algorithm.callback.data['worst']
    y_feasibilities: List[bool] = res.algorithm.callback.data['feasibility']
    x_values = [i for i in range(len(y_values))]
    marker_types = {
        True: 'circle',
        False: 'cross'
    }
    scatters = [
        go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=f'{selection.name}-{crossover.name}',
            marker={
                'symbol': [marker_types[i] for i in y_feasibilities],
            }
        )
    ]
    if pop_dispersion:
        scatters.append(
            go.Scatter(
                x=x_values + x_values[::-1],  # x, then x reversed
                y=np.concatenate((y_worst, y_values[::-1])),  # upper, then lower reversed
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        )

    return scatters


def binary_tournament(pop, P, _, **kwargs):
    # The P input defines the tournaments and competitors
    n_tournaments, n_competitors = P.shape

    if n_competitors != 2:
        raise Exception("Only pressure=2 allowed for binary tournament!")

    # the result this function returns
    S = np.full(n_tournaments, -1, dtype=np.int)

    # now do all the tournaments
    for i in range(n_tournaments):
        a, b = P[i]

        # if the first individual is better, choose it
        if pop[a].F < pop[b].F:
            S[i] = a

        # otherwise take the other individual
        else:
            S[i] = b

    return S


def main():
    problem = Knapsack(
        n_items=N_ITEMS,
        W=WEIGHTS,
        P=PROFITS,
        C=MAXIMUM_CAP
    )
    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True
    )
    selections = [RandomSelection(), TournamentSelection(pressure=2, func_comp=binary_tournament)]
    crossovers = [SBX(), UniformCrossover(prob=1.0), TwoPointCrossover()]

    traces = []
    for selection in selections:
        for crossover in crossovers:
            traces += run_and_return_scatter(
                problem,
                algol,
                selection=selection,
                crossover=crossover,
            )

    combine_and_save_scatter(traces)


if __name__ == "__main__":
    main()
