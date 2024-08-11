import numpy as np
from scipy.interpolate import LinearNDInterpolator

class DiscountCurve:
    def __init__(self, dfs, times, kind='linear'):
        self.dfs = dfs
        self.times = times
        self.interp = np.interp(self.times, self.dfs)

    def __call__(self, x):
        self.interp(x)

class DividentCurve:
    def __init__(self, dfs, times, kind='linear'):
        self.dfs = dfs
        self.times = times
        self.interp = np.interp(self.times, self.dfs)

    def __call__(self, x):
        self.interp(x)