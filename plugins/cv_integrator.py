import mitsuba as mi
import drjit as dr
from typing import Union, Any
from .welford import StatisticsEstimator
from .twostatepath import TwoStatePRBIntegrator
from .twostatevolpath import TwoStatePRBVolpathIntegrator
from utils import runtime

class CVIntegrator(mi.ad.integrators.common.RBIntegrator):
    """
    This integrator encapsulates the control variate logic around an inner
    integrator. It implements the 'path space' variant of the algorithm. This
    means it expects a 'twostate' integrator as inner integrator to render the
    correlated images. For a more general implementation, see metaintegrator.py
    """
    def __init__(self, props):
        super().__init__(props)
        self.integrator = props.get('integrator')
        if type(self.integrator) not in (TwoStatePRBIntegrator, TwoStatePRBVolpathIntegrator):
            raise ValueError("CV integrator expects a 'twostate' integrator as nested integrator !")

        self.adjoint_integrator = props.get('adjoint_integrator', self.integrator)
        self.warmup = props.get('warmup', 1)
        # Statistics for the control variates
        self.beta1 = props.get('beta1', 0.9)
        self.beta2 = props.get('beta2', 0.999)
        self.reset()

    def reset(self):
        self.init = False

    def init_buffers(self):
        # Initialize the statistics estimators if not already done
        if self.init:
            return

        self.img = mi.TensorXf(0.0)
        self.stats = StatisticsEstimator(self.beta1, self.beta2)
        self.v_n = mi.TensorXf(0.0)
        self.w_s = mi.TensorXf(0.0)
        self.F = mi.TensorXf(0.0)
        self.H = mi.TensorXf(0.0)
        self.it = 0

        self.init = True

    def render(self: mi.SamplingIntegrator,
            scene: mi.Scene,
            sensor: int = 0, # WARN: this could also be an object, but we don't support it
            seed: int = 0,
            spp: int = 0,
            develop: bool = True,
            evaluate: bool = True) -> mi.TensorXf:

        assert isinstance(sensor, int)
        self.init_buffers()

        self.H, self.F = self.integrator.render_twostates(scene, sensor, seed, spp)

        # Compute the control weights and update statistics
        if self.it > self.warmup:
            v_0, v_1, cov = self.stats.get()
            self.v_n = self.w_s**2 * (self.v_n + v_0) + v_1 - 2*self.w_s * cov
            dr.schedule(self.v_n)
            self.w_s = cov / (v_0 + self.v_n)
            dr.schedule(self.w_s)
            self.w_s = dr.select(dr.isnan(self.w_s) | dr.isinf(self.w_s), 0.0, self.w_s)
            self.w_s = dr.clamp(self.w_s, 0.0, 1.0)

        self.img = self.w_s * (self.img - self.H) + self.F
        dr.schedule(self.img)

        if self.it > 0:
            self.stats.update(self.H, self.F)

        self.it += 1
        return self.img

    def render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        return self.adjoint_integrator.render_forward(scene, params, sensor, seed, spp)

    def render_backward(self: mi.SamplingIntegrator,
                    scene: mi.Scene,
                    params: Any,
                    grad_in: mi.TensorXf,
                    sensor: Union[int, mi.Sensor] = 0,
                    seed: int = 0,
                    spp: int = 0) -> None:
        return self.adjoint_integrator.render_backward(scene, params, grad_in, sensor, seed, spp)
