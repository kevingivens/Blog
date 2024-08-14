import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

# QL_EPSILON # machine precision, but can also be eps variable

"""
In words,

Dimension of simulation

paths X bins X times

"""


class LocalVolSurface:
    def __init__(self, ref_date: pd.Timestamp, time_grid: ArrayLike, strikes: ArrayLike, L: ArrayLike):
        self.ref_date = ref_date
        self.time_grid = time_grid
        self.strikes = strikes
        self.L = L
        # self.dc = dc   # do I need this?
        X, Y = np.meshgrid(X, Y)
        self._interp2d = LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)

    def __call__(self, *args: np.Any, **kwargs: np.Any) -> np.Any:
        return self._interp2d()



def perform_calculations(time_grid, sim_params, local_vol, lev_func, mixing_factor, heston_process):
    # simulation settings 
    n_paths = sim_params.n_paths
    n_bins = sim_params.n_bins
    x_eps = sim_params.x_eps
    
    heston_process = HestonProcess()
    spot = heston_process.s0()
    v0 = heston_process.v0()
    dc = heston_process.riskFreeRate().dayCounter()
    reference_date = heston_process.riskFreeRate().referenceDate()

    lv0: float = local_vol(np.array([0.0, spot]))/np.sqrt(v0)

    # leverage function or matrix?
    L = np.ones(n_bins, len(time_grid))

    strikes = np.ones(n_bins, time_grid, n_bins)

    # vStrikes(timeGrid_.size())
    
    for i in range(len(time_grid)):
        u = int(n_bins/2)
        # dx = spot*np.sqrt(QL_EPSILON)
        dx = spot*np.sqrt(x_eps)

        for j in range(n_bins):
            strikes[i, j] = spot + (j - u)*dx

        # std::fill(L.column_begin(0), L.column_end(0), lv0)

        # Fixed Local Vol Surface
        lev_func = LocalVolSurface(
            reference_date,
            time_grid
            strikes, 
            L, 
            # dc
        )

        slv_process = HestonSLVProcess(heston_process, lev_func, mixing_factor)

        pairs = np.ones(n_paths, spot, v0)

        k = int(n_paths / n_bins)
        m = n_paths % n_bins

        time_steps = time_grid.size()-1
        
        paths = np.ones(n_paths, time_steps, 2)

        # brownianGenerator = brownianGeneratorFactory_.create(2, time_steps)
    
        rng = np.random.default_rng(seed=42)

        for i in range(n_paths):
            # brownianGenerator.nextPath()
            tmp = np.ones(2)
            for j in range(time_steps):
                # brownianGenerator.nextStep(tmp)
                paths[i][j][0] = tmp[0]
                paths[i][j][1] = tmp[1]

        for n in range(1, time_grid.size()):
            t = time_grid.at(n-1)
            dt = time_grid.dt(n-1)

            # Array x0(2), dw(2)
            x0 = np.ones(2)
            dw = np.ones(2)

            for i in range(n_paths):
                x0[0] = pairs[i].first
                x0[1] = pairs[i].second
                dw[0] = paths[i][n-1][0]
                dw[1] = paths[i][n-1][1]
                x0 = slv_process.evolve(t, x0, dt, dw)
                pairs[i].first = x0[0]
                pairs[i].second = x0[1]

            np.sort(pairs.begin(), pairs.end())

            s = 0 
            e = 0
            for i in range(n_bins):
                inc = int(k + int(i < m))
                e = s + inc
                sum_val=0.0
                for j in range(s, e):
                    sum_val += pairs[j].second
                sum_val/=inc

                strikes[n][i] = 0.5*(pairs[e-1].first + pairs[s].first)
                L[i][n] = np.sqrt(localVol_.localVol(t, strikes[n, i], True))/sum_val
                s = e

            # leverageFunction_.setInterpolationLinear>()