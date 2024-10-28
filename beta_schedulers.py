import numpy as np

class BetaSchedulerSinus():
    def __init__(self, beta_max, period):
        self.beta_max = beta_max
        self.beta = 0.0
        self.period = period

    def __call__(self, step):
        if step % self.period == 0:
            self.beta = 0.0
        else:
            self.beta = 0.5 * ( self.beta_max * np.sin( (step % self.period)/float(self.period) * np.pi - np.pi/2) + self.beta_max )

        return self.beta

class BetaSchedulerSinusAlternating():
    def __init__(self, beta_max, period):
        self.beta_max = beta_max
        self.beta = 0.0
        self.period = period
        self.periodcount = 0

    def __call__(self, step):

        if step > 0 and step % self.period == 0:
            self.periodcount = self.periodcount + 1

        if self.periodcount % 3 == 0:
            self.beta = 0.0

        else:
            if step % (3*self.period) > self.period:
                self.beta = 0.5 * (self.beta_max  + self.beta_max * np.sin(
                - np.pi/2 + np.pi*( (step % (3*self.period)-self.period) % (2*self.period)) /float(2*self.period)
            ))

        return self.beta
