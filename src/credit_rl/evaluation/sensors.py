"""Actor-only measurement errors. No simulator or future information is read."""
import hashlib
import numpy as np


class ObservationSensor:
    def __init__(self, customer_id, seed, *, intercept=0., slope=1., noise=0., lag=0, measurement_sigma=0.):
        if noise < 0 or lag not in (0,1) or measurement_sigma < 0 or slope <= 0:
            raise ValueError("Invalid sensor configuration")
        identity = int.from_bytes(hashlib.blake2b(customer_id.encode(),digest_size=8).digest(),"little")
        self.rng = np.random.default_rng(np.random.SeedSequence([seed,identity]))
        self.intercept,self.slope,self.noise,self.lag,self.measurement_sigma=intercept,slope,noise,lag,measurement_sigma
        self.previous = None

    def __call__(self, observation):
        x = np.asarray(observation,dtype=np.float32).copy()
        current = float(x[10]); value = current if self.previous is None or not self.lag else self.previous
        self.previous = current
        # Preserve exact identity, including endpoints; do not introduce clipping artifacts.
        if self.intercept or self.slope != 1 or self.noise:
            clipped=np.clip(value,1e-7,1-1e-7)
            score=self.intercept+self.slope*np.log(clipped/(1-clipped))+self.noise*self.rng.normal()
            value=float(np.exp(-np.logaddexp(0.,-score)))
        x[10]=value
        if self.measurement_sigma:
            # Correlated balance/utilization reporting error keeps their identity with the true limit.
            for indices in ((2,3),(7,)):
                factor=np.exp(self.measurement_sigma*self.rng.normal()-.5*self.measurement_sigma**2)
                for index in indices:
                    raw=float(x[index])/max(1-float(x[index]),1e-8)
                    x[index]=raw*factor/(1+raw*factor)
        return x
