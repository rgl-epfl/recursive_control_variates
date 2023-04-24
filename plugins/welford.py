import mitsuba as mi
import drjit as dr

class WelfordVarianceEstimator():
    """
    Estimates the variance of a random variable in an online manner, Using
    Welford's online algorithm:
      - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
      - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm

    The estimates use exponential moving averages akin to Adam [Kingma and Ba
    2015] to downweight earlier samples that follow a different distribution in
    an optimization.
    """

    def __init__(self, beta1=0.9, beta2=0.999):
        import mitsuba as mi
        self.n = mi.TensorXi(0)
        dr.make_opaque(self.n)
        self.mean = 0
        self.var = 0
        # EMA weight for the mean
        self.beta1 = mi.TensorXf(beta1)
        # EMA weight for the covariance
        self.beta2 = mi.TensorXf(beta2)

    def update(self, x):
        dx1 = dr.select(self.n > 0, x - self.mean / (1 - self.beta1**self.n), x)
        self.mean = self.beta1 * self.mean + (1 - self.beta1) * x
        self.n += 1
        dx2 = x - self.mean / (1 - self.beta1**self.n)
        self.var = self.beta2 * self.var + (1 - self.beta2) * dx1 * dx2
        dr.schedule(self.mean, self.var, self.n)

    def get(self):
        return self.var / (1 - self.beta2**self.n)

class StatisticsEstimator():
    """
    Estimates the mean, variance and covariance of 2 given random variables,
    using Welford's online algorithm with EWMAs.
    """

    def __init__(self, beta1=0.9, beta2=0.999):
        import mitsuba as mi
        self.n = mi.TensorXi(0)
        dr.make_opaque(self.n)
        self.mean_x = 0
        self.mean_y = 0
        self.var_x = 0
        self.var_y = 0
        self.cov = 0

        # EMA weight for the mean
        self.beta1 = mi.TensorXf(beta1)
        # EMA weight for the (co)variance
        self.beta2 = mi.TensorXf(beta2)

    def update(self, x, y):

        dx1 = dr.select(self.n > 0, x - self.mean_x / (1 - self.beta1**self.n), x)
        dy1 = dr.select(self.n > 0, y - self.mean_y / (1 - self.beta1**self.n), y)

        self.mean_y = self.beta1 * self.mean_y + (1 - self.beta1) * y
        self.mean_x = self.beta1 * self.mean_x + (1 - self.beta1) * x

        self.n += 1
        dx2 = x - self.mean_x / (1 - self.beta1**self.n)
        dy2 = y - self.mean_y / (1 - self.beta1**self.n)

        self.var_x = self.beta2 * self.var_x + (1 - self.beta2) * dx1 * dx2
        self.var_y = self.beta2 * self.var_y + (1 - self.beta2) * dy1 * dy2
        self.cov = self.beta2 * self.cov + (1 - self.beta2) * dx1 * dy2

        dr.schedule(self.mean_x, self.mean_y, self.var_x, self.var_y, self.cov, self.n)

    def get(self):
        return self.var_x / (1 - self.beta2**self.n), self.var_y / (1 - self.beta2**self.n), self.cov / (1 - self.beta2**self.n)
