import numpy as np


class HestonProcess:
    def __init__(self,
                 heston_model: HestonModel,
                 disc_curve: DiscountCurve,
                 div_curve: DividendCurve,
    ) -> None:
        self.disc_curve = risk_free_rate
        self.div_curve = discount_yield
        self.heston_model = heston_model