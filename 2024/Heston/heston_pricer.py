from enum import Enum
from scipy import integrate
import numpy as np


"""

 //! analytic Heston-model engine based on Fourier transform

    /*! Integration detail:
        Two algebraically equivalent formulations of the complex
        logarithm of the Heston model exist. Gatherals [2005]
        (also Duffie, Pan and Singleton [2000], and Schoutens,
        Simons and Tistaert[2004]) version does not cause
        discoutinuities whereas the original version (e.g. Heston [1993])
        needs some sort of "branch correction" to work properly.
        Gatheral's version does also work with adaptive integration
        routines and should be preferred over the original Heston version.
    */

    /*! References:

        Heston, Steven L., 1993. A Closed-Form Solution for Options
        with Stochastic Volatility with Applications to Bond and
        Currency Options.  The review of Financial Studies, Volume 6,
        Issue 2, 327-343.

        A. Sepp, Pricing European-Style Options under Jump Diffusion
        Processes with Stochastic Volatility: Applications of Fourier
        Transform (<http://math.ut.ee/~spartak/papers/stochjumpvols.pdf>)

        R. Lord and C. Kahl, Why the rotation count algorithm works,
        http://papers.ssrn.com/sol3/papers.cfm?abstract_id=921335

        H. Albrecher, P. Mayer, W.Schoutens and J. Tistaert,
        The Little Heston Trap, http://www.schoutens.be/HestonTrap.pdf

        J. Gatheral, The Volatility Surface: A Practitioner's Guide,
        Wiley Finance

        F. Le Floc'h, Fourier Integration and Stochastic Volatility
        Calibration,
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2362968

        L. Andersen, and V. Piterbarg, 2010,
        Interest Rate Modeling, Volume I: Foundations and Vanilla Models,
        Atlantic Financial Press London.

        L. Andersen and M. Lake, 2018
        Robust High-Precision Option Pricing by Fourier Transforms:
        Contour Deformations and Double-Exponential Quadrature,
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3231626


"""

class AnalyticHestonEngine:
    def __init__(self, model) -> None:
       self.model = model
   
    #def chF(self, z, t):
    #     """ char function """
    #     if (self.model.sigma > 1e-6 or self.model.kappa < 1e-8):
    #         return np.exp(self.lnChF(z, t))
    #     else: 
    #         kappa = self.model.kappa
    #         sigma = self.model.sigma
    #         theta = self.model.theta
    #         rho   = self.model.rho
    #          v0    = self.model.v0
    #
    #         kt = kappa*t
    #         ekt = np.exp(kt)
    #         e2kt = np.exp(2*kt)
    #         rho2 = rho*rho
    #         zpi = z + np.array([0 + 1j])
    # 
    #         results =  np.exp(-(((theta - v0 + ekt*((-1 + kt)*theta + v0))*z*zpi)/ekt)/(2.*kappa))
    #
    #         + (np.exp(-(kt) - ((theta - v0 + ekt*((-1 + kt)*theta + v0))*z*zpi)/(2.*ekt*kappa))*rho*(2*theta + kt*theta - v0 - kt*v0 + ekt*((-2 + kt)*theta + v0))*(1.0 - std::complex<Real>(-z.imag(), z.real()))*z*z)/(2.*kappa*kappa)*sigma#
    #
    #        + (np.exp(-2*kt - ((theta - v0 + ekt *((-1 + kt)*theta + v0))*z*zpi)/(2.*ekt*kappa))*z*z*zpi *(-2*rho2*squared(2*theta + kt*theta - v0 -
    #                t*v0 + ekt*((-2 + kt)*theta + v0))
    #              *z*z*zpi + 2*kappa*v0*(-zpi+ e2kt*(zpi + 4*rho2*z) - 2*ekt*(2*rho2*z+ kt*(zpi + rho2*(2 + kt)*z))) + kappa*theta*(zpi + e2kt
    #            *(-5.0*zpi - 24*rho2*z+ 2*kt*(zpi + 4*rho2*z)) +
    #            4*ekt*(zpi + 6*rho2*z + kt*(zpi + rho2*(4 + kt)*z)))))
    #            /(16.*kappa**4))*sigma**2
    #        
    #        return result
    
    def ln_phi(self, z, t):
        """ ln of Heston characteristic function 
        
            adapted from Andersen Lake, Appendix A"""

        kappa = self.model.kappa
        sigma = self.model.sigma
        theta = self.model.theta
        rho   = self.model.rho
        v0    = self.model.v0

        beta = kappa + rho*sigma*np.array([z.imag(), -z.real()*1j])

        D = np.sqrt(beta**2 + (z*z + np.array([-z.imag(), z.real()*1j])*sigma**2))

        g = (beta - D)/(beta + D)

        # reduce cancelation errors, see. L. Andersen and M. Lake
        if beta.real()*D.real() + beta.imag()*D.imag() > 0.0:
            r = -sigma**2*z*(z + np.array([0 + 1j]))/(g+D)
        else:
            r = beta - D

        if (D.real() != 0.0) or (D.imag() != 0.0):
            y = np.expm1(-D*t)/(2.*D)
        else:
            y = -0.5*t

        A = (kappa*theta/sigma**2)*(r*t - 2.0*np.log1p(-r*y))
        B = z*np.array([z.real(), z.imag()+1j])*y/(1.0-r*y)

        return A+v0*B
    

class ComplexLogFormula(Enum):
    Gatheral  = 1 # Gatheral form of characteristic function w/o control variate
    BranchCorrection = 2 # old branch correction form of the characteristic function w/o control variate
    AndersenPiterbarg = 3 # Gatheral form with Andersen-Piterbarg control variates
    AndersenPiterbargOptCV = 4  # same as AndersenPiterbarg, but a slightly better control variate
    # Gatheral form with asymptotic expansion of the characteristic function as control variate
    # https://hpcquantlib.wordpress.com/2020/08/30/a-novel-control-variate-for-the-heston-model
    AsymptoticChF = 5
    # angled contour shift integral with control variate
    AngledContour = 6
    # angled contour shift integral w/o control variate
    AngledContourNoCV = 6
    # auto selection of best control variate algorithm from above
    OptimalCV = 7


class AP_Helper:
    def __init__(
        self,
        term, 
        fwd, 
        strike, 
        cpx_log: ComplexLogFormula,
        engine: AnalyticHestonEngine,
        alpha
    ): 
        self.term = term
        self.fwd = fwd
        self.strike = strike
        self.freq = np.log(fwd/strike)
        self.cpx_log = cpx_log
        self.engine = engine
        self.alpha = alpha
        self.s_alpha = np.exp(alpha*self.freq)
        
        # Model parameters
        v0    = engine.model.v0
        kappa = engine.model.kappa
        theta = engine.model.theta
        sigma = engine.model.sigma
        rho   = engine.model.rho

        match self.cpx_log:
          case ComplexLogFormula.AndersenPiterbarg:
            vAvg_ = (1-np.exp(-kappa*term))*(v0 - theta)/(kappa*term) + theta
            # break
          case ComplexLogFormula.AndersenPiterbargOptCV:
            vAvg_ = -8.0*np.log(engine.chF(np.array([0, self.alpha*1j]), term).real())/term
            # break
          case ComplexLogFormula.AsymptoticChF:
            phi_ = -(v0+term*kappa*theta)/sigma * np.array([np.sqrt(1-rho*rho), rho*1j])

            psi_ = np.array([
               (kappa- 0.5*rho*sigma)*(v0 + term*kappa*theta) + kappa*theta*np.log(4*(1-rho*rho)),
                - ((0.5*rho*rho*sigma - kappa*rho)/np.sqrt(1-rho*rho)*(v0 + kappa*theta*term)- 
                   2*kappa*theta*np.atan(rho/np.sqrt(1-rho*rho)))])/(sigma**2)
          
          case ComplexLogFormula.AngledContour:
            vAvg_ = (1-np.exp(-kappa*term))*(v0 - theta) /(kappa*term) + theta
          case ComplexLogFormula.AngledContourNoCV:
            r = rho - sigma*self.freq / (v0 + kappa*theta*term)
            tanPhi_ = np.atan((r*self.freq < 0.0)? np.pi/12*boost::math::sign(self.freq) : 0.0)
            # break 
          case _:
            raise ValueError("unknown control variate")
