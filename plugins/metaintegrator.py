import mitsuba as mi
import drjit as dr
from typing import Union, Any
from .welford import StatisticsEstimator
from utils import runtime

class MetaIntegrator(mi.ad.integrators.common.RBIntegrator):
    """
    This integrator is a meta integrator that can be used to wrap other integrators and add
    control variates to them. It can also be used to denoise the output of the wrapped integrator.
    """
    def __init__(self, props):
        super().__init__(props)
        self.denoise = props.get('denoise', False)
        self.force_baseline = props.get('force_baseline', False)
        self.integrator = props.get('integrator')
        self.adjoint_integrator = props.get('adjoint_integrator', self.integrator)
        self.aov_integrator = mi.load_dict({
            'type': 'aov',
            'aovs': 'albedo:albedo,normals:sh_normal',
            'integrator': self.integrator
        })
        # Placeholders for denoiser and sensor-to-world transform
        self.denoiser = None
        self.to_sensor = None

        self.method = props.get('method', 'baseline')
        self.pre_update = props.get('pre_update', False)
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

        if 'cv' not in self.method:
            return

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

        if self.method == 'cv_ps':
            self.H, self.F = self.integrator.render_twostates(scene, sensor, seed, spp)

        if self.denoise:
            # If not already done, compute the sensor-to-world transform
            # WARN: this does not work if the sensor is animated
            if self.to_sensor is None:
                self.to_sensor = scene.sensors()[sensor].world_transform().inverse()

            # Render with AOV integrator
            aovs = self.aov_integrator.render(scene, sensor, seed, spp, develop, evaluate)
            img_noisy = aovs[..., :3]
            albedo = aovs[..., 3:6]
            normals = aovs[..., 6:9]

            # Initialize denoiser
            if self.denoiser is None:
                self.denoiser = mi.OptixDenoiser(aovs.shape[:2], albedo=True, normals=True, temporal=False)

        elif self.method != 'cv_ps':
            img_noisy = self.integrator.render(scene, sensor, seed, spp, develop, evaluate)

        if self.method == 'cv_pss':
            self.F = img_noisy

        if 'cv' in self.method:
            # Compute the control weights and update statistics
            if self.it > 0 and self.pre_update:
                self.stats.update(self.H, self.F)

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

            if self.it > 0 and not self.pre_update:
                self.stats.update(self.H, self.F)

            self.it += 1
        else:
            self.img = img_noisy

        if self.method == 'cv_pss':
            # Re render the current state with the seed for the next iteration
            self.H = self.integrator.render(scene, sensor, seed+1, spp, develop, evaluate)

        if self.denoise:
            # Denoise
            return self.denoiser(self.img, albedo=albedo, normals=normals, to_sensor=self.to_sensor)
        elif self.force_baseline:
            # Return the noisy image. This is only so we can look at images with the different methods, while still taking the same steps.
            if self.method != 'cv_ps':
                return img_noisy
            return self.integrator.render(scene, sensor, seed, spp, develop, evaluate)
        else:
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
