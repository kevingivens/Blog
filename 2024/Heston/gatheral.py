import numpy as np
from scipy.integrate import quad


def phi(z, kappa, sigma, rho, theta, v0, t):
    """  following Andersen """
    beta = kappa - sigma*rho*z*1j
    d = np.sqrt(beta**2 + sigma**2*z*(z + 1j))
    g = (beta - d) / (beta + d)
    a = (kappa*theta)/(sigma**2)*((beta - d)*t - 2*np.log((1 - g*np.exp(-d*t))/(1 - g)))
    b = (beta - d)/(sigma**2)*(1 - np.exp(-d*t))/(1 - g*np.exp(-d*t))
    return np.exp(a + v0*b)


def heston_phi(z, kappa, sigma, rho, theta, v0, t):

    beta = kappa - sigma*rho*z*1j
    d = np.sqrt(beta**2 + sigma**2*z*(z + 1j))
    g = (beta - d)/(beta + d)

    # reduce cancelation errors, see. L. Andersen and M. Lake
    if beta.real*d.real + beta.imag*d.imag > 0.0:
        r = -sigma**2*z*(z + 1j)/(g+d)
    else:
        r = beta - d

    if (d.real != 0.0) or (d.imag != 0.0):
        y = np.expm1(-d*t)/(2.*d)
    else:
        y = -0.5*t

    a = (kappa*theta/sigma**2)*(r*t - 2.0*np.log1p(-r*y))
    b = z*(z + 1j)*y/(1.-r*y)

    return a+v0*b


def pv(rho, sigma, v0, kappa, theta, t):
    c_inf = np.min(0.2, np.max(0.0001, np.sqrt(1.0-rho*rho)/sigma))*(v0 + kappa*theta*t)
    fj_helper = Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, t, strike_price, ratio, 1)
    p1 = quad(Fj_Helper, )
            
    # p1 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 1))/np.pi
            
    nevals += integration.numberOfEvaluations()

    # p2 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 2))/np.pi
            
    nevals += integration.numberOfEvaluations()

    value = spot_price*dividend_discount*(p1 + int(option_type)*0.5) - strike_price*risk_free_discount*(p2 + int(option_type)*0.5)


"""
def fj(phi):
    # rsigma_
    # t0
    # sigma
    # term  t?
    # v0
    # ratio
    # j  (1 or 0)
    x = np.log(s0)
    sx = np.log(strike)
    dd = x-np.log(ratio)
    rsigma = rho * sigma
    t0 = kappa if j ==1 else kappa - rho*sigma
    # self.engine = engine
    # log branch counter
    b = 0        # log branch counter
    g_km1 = 0    # imag part of last log value
    # rpsig = rho * sigma *phi
    t1 = t0 + np.array([0 - rho*sigma*phi*1j])
    d = np.sqrt(t1*t1 - sigma**2*phi*np.array([-phi + 1j if j== 1 else -1j]))  # this is stupid
    ex = np.exp(-d*term)
    add_on_term = self.engine.addOnTerm(phi, term, j) if self.engine else 0.

    if phi != 0.0:
        if sigma > 1e-5:
            p = (t1-d)/(t1+d)
            g = np.log((1.0 - p*ex)/(1.0 - p))
            result =  np.exp(
                   v0*(t1-d)*(1.0-ex)/(sigma**2*(1.0-ex*p)) 
                   + (kappa*theta)/sigma**2*((t1-d)*term-2.0*g) 
                   + np.array([0.0 + phi*(dd-sx)*1j]) 
                   + add_on_term
                   ).imag()/phi
            return result
        else:
            td = phi/(2.0*t1) * np.array([-phi (j_== 1)? 1 : -1])
            p = td*sigma**2/(t1+d)
            g = p*(1.0-ex)

            return np.exp(
                  v0*td*(1.0-ex)/(1.0-p*ex) 
                  + (kappa*theta)*(td*term-2.0*g/sigma**2)
                  + np.array([0.0, phi*(dd-sx)*1j])
                  + add_on_term
                  ).imag()/phi
    else:
        # use l'Hospital's rule to get lim_{phi->0}
        if j == 1:
            kmr = rsigma - kappa
            if np.abs(kmr) > 1e-7:
                return dd-sx+ (np.exp(kmr*term_)*kappa_*theta_-kappa_*theta_*(kmr*term_+1.0) ) / (2*kmr*kmr) - v0_*(1.0-np.exp(kmr*term_)) / (2.0*kmr)        
            else:
                # \kappa = \rho * \sigma
                return dd - sx + 0.25*kappa*theta*term**2+ 0.5*v0*term
        else:
            return dd_-sx_- (
                 np.exp(-kappa_*term_)*kappa_*theta_ +kappa_*theta_*(kappa_*term_-1.0))/(2*kappa_*kappa_)- v0_*(1.0-np.exp(-kappa_*term_))/(2*kappa_)


c_inf = np.min(0.2, np.max(0.0001, np.sqrt(1.0-rho*rho)/sigma))*(v0 + kappa*theta*term)
fj_helper = Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 1)
p1 = quad(Fj_Helper, )
            
# p1 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 1))/np.pi
            
nevals += integration.numberOfEvaluations()

p2 = integration.calculate(c_inf, Fj_Helper(kappa, theta, sigma, v0, spot_price, rho, engine, cpx_log, term, strike_price, ratio, 2))/np.pi
            
nevals += integration.numberOfEvaluations()

value = spot_price*dividend_discount*(p1 + int(option_type)*0.5) - strike_price*risk_free_discount*(p2 + int(option_type)*0.5)
"""