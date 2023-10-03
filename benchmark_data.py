import datetime
import logging
import time
import numpy as np
import plotly.offline as pyo
import plotly.graph_objs as go

from dataclasses import dataclass
from typing import List, Tuple

from numpy import ndarray
from pymoo.algorithms.soo.nonconvex import ga
from pymoo.operators.crossover.pntx import TwoPointCrossover, PointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.optimize import minimize

from knapsack import BestCandidateCallback, combine_and_save_scatter
from model_params import params_100r_140c, params_750c_3250r
from nrp_pymoo import NRP


@dataclass
class ScriptParameters:
    pop_size: int = 100
    termination_n_gen = 200
    selections = [RandomSelection()]
    crossovers = [PointCrossover(n_points=4), SBX()]
    n_iters = 5


@dataclass
class RunData:
    y_best: List[float]
    y_feasibility: List[bool]
    time_sec: float


@dataclass
class GraphColor:
    primary: str
    fill: str


colors = [
    GraphColor(primary='rgb(0,100,80)', fill='rgba(0,100,80,0.2)'),
    GraphColor(primary='rgb(100,100,80)', fill='rgba(100,100,80,0.2)')
]


def get_average_and_percentiles(data: List[RunData]) -> Tuple[ndarray, ndarray, ndarray]:
    y_values = [d.y_best for d in data]
    bests = np.column_stack(y_values)
    means = np.mean(bests, axis=1)
    percentiles_25 = np.percentile(bests, 25, axis=1)
    percentiles_75 = np.percentile(bests, 75, axis=1)
    return percentiles_25, means, percentiles_75


def plot_run_data_with_error_bars(data: List[RunData], graph_name=None, color: GraphColor = None):
    y_lower, y_mean, y_upper = get_average_and_percentiles(data)
    x = [i for i in range(0, len(list(y_lower)))]
    scatters = [
        go.Scatter(
            x=x,
            y=y_mean,
            line=dict(color=color.primary or 'rgb(0,100,80)'),
            mode='lines',
            name=graph_name or ''
        ),
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=np.concatenate((y_upper, y_lower[::-1])),  # upper, then lower reversed
            fill='toself',
            fillcolor=color.fill or 'rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        )
    ]
    return scatters
    # pyo.plot(fig, filename='run_data_w_error_bars.html')


def main():
    script_params = ScriptParameters()
    params = params_750c_3250r()
    nrp = NRP(params)
    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        pop_size=script_params.pop_size
    )

    figs = []
    j = 0
    for selection in script_params.selections:
        for crossover in script_params.crossovers:
            data = []
            for i in range(script_params.n_iters):
                print(
                    f'{datetime.datetime.now()} Running iteration {selection.name} {crossover.name} {i} '
                    f'of {script_params.n_iters}')
                t0 = time.perf_counter()
                res = minimize(
                    problem=nrp,
                    algorithm=algol,
                    termination=('n_gen', script_params.termination_n_gen),
                    verbose=False,
                    callback=BestCandidateCallback(),
                    selection=selection,
                    crossover=crossover
                )
                t1 = time.perf_counter()
                data.append(RunData(
                    y_best=res.algorithm.callback.data['best'],
                    y_feasibility=res.algorithm.callback.data['feasibility'],
                    time_sec=t1 - t0
                ))
            figs += plot_run_data_with_error_bars(data, graph_name=f'{selection.name}-{crossover.name}',
                                                  color=colors[j])
            j += 1
    combine_and_save_scatter(figs, 'errorbars.html', params.fo_optimum)


if __name__ == "__main__":
    main()
