from enum import Enum, IntEnum
from scipy.integrate import quad
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

class OptionType(IntEnum):
    PUT = -1
    CALL = 1


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
            tanPhi_ = np.atan(np.where(r*self.freq < 0.0, np.pi/12*np.sign(self.freq),  0.0))
            # break 
          case _:
            raise ValueError("unknown control variate")
           
    def __call__(self, u) -> np.Any:
        # assert self.engine.addOnTerm(u, term_, 1) == std::complex<Real>(0.0) and
        #       self.engine.addOnTerm(u, term_, 2) == std::complex<Real>(0.0),
        #            "only Heston model is supported"

        std::complex<double> i(0, 1)

        if (self.cpx_log == ComplexLogFormula.AngledContour) or (self.cpx_log == ComplexLogFormula.AngledContourNoCV) or (self.cpx_log == AsymptoticChF):
            std::complex<Real> h_u(u, u*tanPhi_ - alpha_)
            std::complex<Real> hPrime(h_u-i)

            std::complex<Real> phiBS(0.0)
            if self.cpx_log == ComplexLogFormula.AngledContour:
                phiBS = np.exp(-0.5*vAvg_*term_*(hPrime*hPrime + std::complex<Real>(-hPrime.imag(), hPrime.real())))
            elif sel.cpx_log == ComplexLogFormula.AsymptoticChF:
                phiBS = np.exp(u*std::complex<Real>(1, tanPhi_)*phi_ + psi_)
            return np.exp(-u*tanPhi_*freq_)*(np.exp(np.array([0.0 + u*freq_*1j])))*std::complex<Real>(1, tanPhi_)
                      *(phiBS - self.engine.chF(hPrime, term_))/(h_u*hPrime)
                      ).real()*s_alpha_
        elif (self.cpx_log == ComplexLogFormula.AndersenPiterbarg) or (self.cpx_log == ComplexLogFormula.AndersenPiterbargOptCV):
            std::complex<Real> z(u, -alpha_)
            std::complex<Real> zPrime(u, -alpha_-1)
            std::complex<Real> phiBS = np.exp(-0.5*vAvg_*term_*(zPrime*zPrime +
                        std::complex<Real>(-zPrime.imag(), zPrime.real()))
            )
            return (np.exp(std::complex<Real> (0.0, u*freq_)) * (phiBS - self.engine.chF(zPrime, term_)) / (z*zPrime)).real()*s_alpha_
        else:
            raise ValueError("unknown control variate")


class Fj_Helper:
    def __init__(
        self,
        kappa, 
        theta,
        sigma, 
        v0, 
        s0, 
        rho,
        cpx_log: ComplexLogFormula,
        term, 
        strike, 
        ratio, 
        j: int,
        engine = None,
    ):
        self.j = j
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.v0 = v0
        self.cpx_log = cpx_log
        self.term = term
        self.x = np.log(s0)
        self.sx = np.log(strike)
        self.dd = self.x-np.log(ratio)
        # self.sigma2 = self.sigma_*self.sigma_
        self.rsigma = rho*self.sigma
        self.t0 = kappa if j==1 else kappa - rho*sigma
        self.engine = engine
        # log branch counter
        self.b = 0        # log branch counter
        self.g_km1 = 0  # imag part of last log value

    def __call__(self, phi: float):
        rpsig = rsigma_*phi
        t1 = self.t0 + np.array([0 -rpsig*1j])
        d = np.sqrt(t1*t1 - self.sigma**2*phi*std::complex<Real>(-phi, (j_== 1)? 1 : -1))
        ex = np.exp(-d*self.term)
        add_on_term = self.engine.addOnTerm(phi, self.term, self.j) if self.engine else 0.

        if self.cpx_log == ComplexLogFormula.Gatheral:
            if phi != 0.0:
                if self.sigma > 1e-5:
                    p = (t1-d)/(t1+d)
                    g = np.log((1.0 - p*ex)/(1.0 - p))

                    return np.exp(self.v0*(t1-d)*(1.0-ex)/(self.sigma**2*(1.0-ex*p))
                                 + (self.kappa*self.theta)/self.sigma**2*((t1-d)*self.term-2.0*g)
                                 + np.array([0.0 + phi*(self.dd-self.sx)*1j])
                                 + add_on_term
                                 ).imag()/phi
                else:
                    td = phi/(2.0*t1) * std::complex<Real>(-phi, (j_== 1)? 1 : -1)
                    p = td*self.sigma**2/(t1+d)
                    g = p*(1.0-ex)

                    return np.exp(self.v0*td*(1.0-ex)/(1.0-p*ex)
                                 + (self.kappa*self.theta)*(td*self.term-2.0*g/self.sigma**2)
                                 + np.array([0.0, phi*(self.dd-self.sx)*1j])
                                 + add_on_term
                                 ).imag()/phi
        
            else:
                # use l'Hospital's rule to get lim_{phi->0}
                if self.j == 1:
                    kmr = self.rsigma - self.kappa
                    if np.abs(kmr) > 1e-7:
                        return dd_-sx_
                            + (np.exp(kmr*term_)*kappa_*theta_
                               -kappa_*theta_*(kmr*term_+1.0) ) / (2*kmr*kmr)
                            - v0_*(1.0-np.exp(kmr*term_)) / (2.0*kmr)
                    
                    else:
                        # \kappa = \rho * \sigma
                        return dd_-sx_ + 0.25*kappa_*theta_*term_*term_
                                       + 0.5*v0_*term_;
                
                else:
                    return dd_-sx_
                        - (np.exp(-kappa_*term_)*kappa_*theta_
                           +kappa_*theta_*(kappa_*term_-1.0))/(2*kappa_*kappa_)
                        - v0_*(1.0-np.exp(-kappa_*term_))/(2*kappa_)
                
            
        elif self.cpx_log == ComplexLogFormula.BranchCorrection:
            p = (t1+d)/(t1-d)

            # next term: g = std::log((1.0 - p*std::exp(d*term_))/(1.0 - p))

            # the exp of the following expression is needed.
            e = np.log(p)+d*self.term

            # does it fit to the machine precision?
            if np.exp(-e.real()) > QL_EPSILON:
                g = np.log((1.0 - p/ex)/(1.0 - p))
            else:
                # use a "big phi" approximation
                g = d*self.term + np.log(p/(p - 1.0))

                if (g.imag() > np.pi) or (g.imag() <= -np.pi):
                    # get back to principal branch of the complex logarithm
                    Real im = std::fmod(g.imag(), 2*np.pi)
                    if im > np.pi:
                        im -= 2*np.pi
                    elif im <= -np.pi:
                        im += 2*np.pi

                    g = std::complex<Real>(g.real(), im)
                

            # be careful here as we have to use a log branch correction
            # to deal with the discontinuities of the complex logarithm.
            # the principal branch is not always the correct one.
            # (s. A. Sepp, chapter 4)
            # remark: there is still the change that we miss a branch
            # if the order of the integration is not high enough.
            tmp = g.imag() - self.g_km1
            if tmp <= -np.pi:
                self.b += self.b
            elif tmp > np.pi:
                self.b -= self.b

            self.g_km1 = g.imag()
            g += 2*b_*np.pi*j

            return np.exp(v0_*(t1+d)*(ex-1.0)/(sigma2_*(ex-p))
                            + (kappa_*theta_)/sigma2_*((t1+d)*term_-2.0*g)
                            + std::complex<Real>(0,phi*(dd_-sx_))
                            + add_on_term
                            ).imag/phi
        else:
            raise ValueError("unknown complex logarithm formula")







def pv(
      risk_free_discount, 
      dividend_discount, 
        spot_price,
        strike_price,
        term,
        kappa, 
        theta,
        sigma,
        v0, 
        rho,
        payoff_type : TypePayoff,
        integration: Integration,
        cpx_log: ComplexLogFormula,
        engine: AnalyticHestonEngine,
        option_type,
):
    """ price a Euro Option using the Heston Model """
    
    ratio = risk_free_discount / dividend_discount

    evaluations = 0

    match cpx_log:
        case ComplexLogFormula.Gatheral | ComplexLogFormula.BranchCorrection:
            c_inf = np.min(0.2, np.max(0.0001, np.sqrt(1.0-rho*rho)/sigma))*(v0 + kappa*theta*term)
            fj_helper = Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 1)
            p1 = quad(Fj_Helper, )
            
            # p1 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 1))/np.pi
            
            evaluations += integration.numberOfEvaluations()

            p2 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 2))/np.pi
            
            evaluations += integration.numberOfEvaluations()

            value = spot_price*dividend_discount*(p1 + int(option_type)*0.5) - strike_price*risk_free_discount*(p2 + int(option_type)*0.5)
            
        case ComplexLogFormula.AndersenPiterbarg | ComplexLogFormula.AndersenPiterbargOptCV | ComplexLogFormula.AsymptoticChF| ComplexLogFormula.OptimalCV:
            c_inf = np.sqrt(1.0-rho*rho)*(v0 + kappa*theta*term)/sigma

            fwd_price = spot_price / ratio

            epsilon = engine.andersenPiterbargEpsilon_*np.pi/(np.sqrt(strike_price*fwd_price)*risk_free_discount)

            def uM():
                return andersenPiterbargIntegrationLimit(c_inf, epsilon, v0, term)


            cv_helper = AP_Helper(
               term, 
               fwd_price, 
               strike_price,
               optimalControlVariate(term, v0, kappa, theta, sigma, rho) if cpx_log == ComplexLogFormula.OptimalCV else cpx_log,
               engine
            )

            cv_value = cv_helper.controlVariateValue()

            h_cv = integration.calculate(c_inf, cv_helper, uM) * np.sqrt(strike_price * fwd_price)/np.pi
            
            evaluations += integration.numberOfEvaluations()

            match option_type:
               case OptionType.Call:
                  value = (cvValue + h_cv)*risk_free_discount
                  # break
               case OptionType.Put:
                  value = (cvValue + h_cv - (fwd_price - strike_price))*risk_free_discount
                  # break
               case _:
                  raise ValueError("unknown option type")
                  break

        case  _:
            raise ValueError("unknown complex log formula")