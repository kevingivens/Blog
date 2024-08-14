import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import LinearNDInterpolator


class DiscountCurve:
    def __init__(self, dfs: ArrayLike, times: ArrayLike, kind='linear'):
        self.dfs = dfs
        self.times = times
        self._interp1d = np.interp(self.times, self.dfs)

    def __call__(self, x):
        self._interp1d(x)
    
        
class BlackVarianceSurface:
    def __init__(self, strikes: ArrayLike, times: ArrayLike, vols: ArrayLike):
        """ strikes 1d array
            times 1d array
            vols: 2d array 
        """
        self.strikes = strikes
        self.times = times
        self.vols = vols
        self._interp2d = LinearNDInterpolator(points, values, fill_value=np.nan, rescale=False)

    def __call__(self, pnts) -> NDArray:
        """
        pnts: 2d array
        """
        self._interp2d(pnts)


class LocalVolatility:
    def __init__(
            self,
            black_var_surface: BlackVarianceSurface, 
            risk_free_curve: DiscountCurve, 
            dividend_curve: DiscountCurve
    ) -> None:
        self.black_var_surface = black_var_surface
        self.risk_free_disc_curve = risk_free_curve
        self.dividend_disc_curve = dividend_curve
        # self.forwards = forwards
        # self.dt = np.min(0.0001, self.times/2.0)
        # self.y = np.log(k/forwards)

    def __call__(self, pnts) -> NDArray:
        # QL signature is (t, y)
        """
        pnts: 2d array
        """
        y = np.log(k/f)
        dy = np.where(np.abs(y) > 0.001, y*0.0001, 0.000001)
        strikep = k * np.exp(dy)
        strikem = k / np.exp(dy)
        w  = self.black_var_surface(t, k)
        wp = self.black_var_surface(t, strikep)
        wm = self.black_var_surface(t, strikem)
    
        # derivatives in strike dimension
        dwdy = (wp-wm)/(2.0*dy)
        d2wdy2 = (wp-2.0*w+wm)/(dy*dy)

        # derivatives in time dimension
        dt = np.min(0.0001, t/2.0)
        drpt = self.risk_free_disc_curve(t+dt, True)  # TODO: check if this means extrapolate
        drmt = self.risk_free_disc_curve(t-dt, True)
        dqpt = self.dividend_disc_curve(t+dt, True)
        dqmt = self.dividend_disc_curve(t-dt, True)
            
        strikept = k*dr*dqpt/(drpt*dq)
        strikemt = k*dr*dqmt/(drmt*dq)
            
        wpt = self.black_var_surface(t+dt, strikept, True)
        wmt = self.black_var_surface(t-dt, strikemt, True)

        assert wpt>=w, f"decreasing variance at strike {k} between time {t} and time {t+dt}"
        assert w>=wmt, f"decreasing variance at strike {k} between time {t-dt}  and time {t}"
         
        dwdt = (wpt-wmt)/(2.0*dt)
    
        den1 = 1.0 - y/w*dwdy
        den2 = 0.25*(-0.25 - 1.0/w + y*y/w/w)*dwdy*dwdy
        den3 = 0.5*d2wdy2
        den = den1 + den2 + den3
        result = dwdt / den

        msg = f"negative local vol^2 at strike {k} and time {t}"
        msg += f" the black vol surface is not smooth enough"
        assert result>=0.0, msg
        return np.sqrt(result)