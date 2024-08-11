from dataclasses import dataclass
import dataclasses
import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

@dataclass
class HestonModel:
    kappa: float
    theta: float
    sigma: float
    rho: float
    v0: float
    """
        arguments_[0] = ConstantParameter(process->theta(),
                                          PositiveConstraint());
        arguments_[1] = ConstantParameter(process->kappa(),
                                          PositiveConstraint());
        arguments_[2] = ConstantParameter(process->sigma(),
                                          PositiveConstraint());
        arguments_[3] = ConstantParameter(process->rho(),
                                          BoundaryConstraint(-1.0, 1.0));
        arguments_[4] = ConstantParameter(process->v0(),
    FellerConstraint : public Constraint {
      private:
        class Impl final : public Constraint::Impl {
          public:
            bool test(const Array& params) const override {
                const Real theta = params[0];
                const Real kappa = params[1];
                const Real sigma = params[2];

                return (sigma >= 0.0 && sigma*sigma < 2.0*kappa*theta);
    """

class YieldTermStructure:
    def __init__(self) -> None:
        pass

    def discount_factors(self, dates: ArrayLike):
        """ compute the discount factor """
        pass

    def forward_rate(self, dates: ArrayLike):
        pass

    def zero_rate(self, dates: ArrayLike):
        pass



class HestonSLVProcess:
    def __init__(self,
                 heston_model: HestonModel,
                 disc_curve: DiscountCurve,
                 div_curve: DividendCurve,
                 lev_func: LeverageFunction,
                 mixing_factor: float
    ) -> None:
        self.disc_curve = risk_free_rate
        self.div_curve = discount_yield
        self.lev_func = lev_func
        self.heston_model = heston_model
        self.mixing_factor = mixing_factor

    def evolve(self, t0, x0: ArrayLike, dt, dw: ArrayLike):

        kappa, theta, rho, sigma = dataclasses.astuple(self.heston_model)
        mixed_sigma = self.mixing_factor * sigma
    
        ret_val = np.array(2)

        ex = np.exp(-kappa*dt)

        m  =  theta + (x0[1]-theta)*np.exp(-kappa*dt)
        s2 =  x0[1]*mixed_sigma**2*ex/kappa*(1-ex) + theta*mixed_sigma**2/(2*kappa)*(1-ex)*(1-ex)
        psi = s2/(m*m)

        if psi < 1.5:
            b2 = 2/psi-1+np.sqrt(2/psi*(2/psi-1))
            b  = np.sqrt(b2)
            a  = m/(1+b2)
            ret_val[1] = a*(b+dw[1])*(b+dw[1])
        else:
            p = (psi-1)/(psi+1)
            beta = (1-p)/m
            u = norm.cdf(dw[1])
            ret_val[1] = 0. if u <= p else np.log((1-p)/(1-u))/beta

            mu = riskFreeRate()->forwardRate(t0, t0+dt, Continuous).rate()
                  - dividendYield()->forwardRate(t0, t0+dt, Continuous).rate()

            rho1 = np.sqrt(1-rho**2)

            l_0: float = self.lev_func.localVol(t0, x0[0], True)
        
            v_0 = 0.5*(x0[1]+ret_val[1])*l_0*l_0

            ret_val[0] = x0[0]*np.exp(
                mu*dt - 0.5*v_0*dt+ rho/mixed_sigma*l_0 *(ret_val[1] - kappa*theta*dt + 0.5*(x0[1]+ret_val[1])*kappa*dt - x0[1]) 
                + rho1*np.sqrt(v_0*dt)*dw[0]
            )

            return ret_val