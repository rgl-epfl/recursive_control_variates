from __future__ import annotations as __annotations__ # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr
import gc

mis_weight = mi.ad.common.mis_weight
from .twostatebsdf import TwoStateBSDF
from .twostatemedium import TwoStateMedium

class _TwoStateRenderOp(dr.CustomOp):
    """
    This class is an implementation detail of the render() function. It
    realizes a CustomOp that provides evaluation, and forward/reverse-mode
    differentiation callbacks that will be invoked as needed (e.g. when a
    rendering operation is encountered by an AD graph traversal).
    """

    def __init__(self) -> None:
        super().__init__()
        self.variant = mi.variant()

    def eval(self, integrator, scene, sensor, params, seed, spp):
        self.scene = scene
        self.sensor = sensor
        self.params = params
        self.integrator = integrator
        self.seed = seed
        self.spp = spp

        with dr.suspend_grad():
            return self.integrator.render_twostates(
                scene=self.scene,
                sensor=sensor,
                seed=seed[0],
                spp=spp[0]
            )

    def forward(self):
        mi.set_variant(self.variant)
        if not isinstance(self.params, mi.SceneParameters):
            raise Exception('An instance of mi.SceneParameter containing the '
                            'scene parameter to be differentiated should be '
                            'provided to mi.render() if forward derivatives are '
                            'desired!')
        self.set_grad_out(0,
            self.integrator.render_forward(self.scene, self.params, self.sensor,
                                           self.seed[1], self.spp[1]))

    def backward(self):
        mi.set_variant(self.variant)
        if not isinstance(self.params, mi.SceneParameters):
            raise Exception('An instance of mi.SceneParameter containing the '
                            'scene parameter to be differentiated should be '
                            'provided to mi.render() if backward derivatives are '
                            'desired!')
        self.integrator.render_backward(self.scene, self.params, self.grad_out()[1],
                                        self.sensor, self.seed[1], self.spp[1])

    def name(self):
        return "TwoStateRenderOp"

def two_state_render(scene: mi.Scene,
                     integrator,
                     params: Any = None,
                     sensor: Union[int, mi.Sensor] = 0,
                     seed: int = 0,
                     seed_grad: int = 0,
                     spp: int = 0,
                     spp_grad: int = 0) -> mi.TensorXf:

    if params is not None and not isinstance(params, mi.SceneParameters):
        raise Exception('params should be an instance of mi.SceneParameter!')

    assert isinstance(scene, mi.Scene)

    if isinstance(sensor, int):
        if len(scene.sensors()) == 0:
            raise Exception('No sensor specified! Add a sensor in the scene '
                            'description or provide a sensor directly as argument.')
        sensor = scene.sensors()[sensor]

    assert isinstance(sensor, mi.Sensor)

    if spp_grad == 0:
        spp_grad = spp

    if seed_grad == 0:
        # Compute a seed that de-correlates the primal and differential phase
        seed_grad = mi.sample_tea_32(seed, 1)[0]
    elif seed_grad == seed:
        raise Exception('The primal and differential seed should be different '
                        'to ensure unbiased gradient computation!')

    return dr.custom(_TwoStateRenderOp, integrator,
                      scene, sensor, params, (seed, seed_grad), (spp, spp_grad))

class TwoStatePRBIntegrator(mi.ad.integrators.prb.PRBIntegrator):
    r"""
    .. _integrator-prb:

    Path Replay Backpropagation (:monosp:`prb`)
    -------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: 6)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

    This plugin implements a basic Path Replay Backpropagation (PRB) integrator
    with the following properties:

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - No reparameterization. This means that the integrator cannot be used for
      shape optimization (it will return incorrect/biased gradients for
      geometric parameters like vertex positions.)

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    See ``prb_basic.py`` for an even more reduced implementation that removes
    the first two features.

    See the papers :cite:`Vicini2021` and :cite:`Zeltner2021MonteCarlo`
    for details on PRB, attached/detached sampling, and reparameterizations.

    .. tabs::

        .. code-tab:: python

            'type': 'prb',
            'max_depth': 8
    """
    def __init__(self, props):
        super().__init__(props)
        self.xyz = props.get("xyz", False)
        self.is_checked = False

    def check_scene(self, scene):
        if self.is_checked:
            return

        self.is_checked = True
        has_twostate = False
        for shape in scene.shapes():
            has_twostate = has_twostate or isinstance(shape.bsdf(), TwoStateBSDF)

            for medium in [shape.interior_medium(), shape.exterior_medium()]:
                if isinstance(medium, TwoStateMedium):
                    raise ValueError("TwoStateMedium is not supported in two state prb integrator!")

        if not has_twostate:
            raise RuntimeError("No TwoStateBSDF found in the scene!")

    def develop(self, sensor, ray, weight, pos, spp, L, alpha):
        # Prepare an ImageBlock as specified by the film
        block = sensor.film().create_block()

        # Only use the coalescing feature when rendering enough samples
        block.set_coalesce(block.coalesce() and spp >= 4)
        block.put(pos, ray.wavelengths, L * weight, alpha)
        # Perform the weight division and return an image tensor
        sensor.film().put_block(block)
        return sensor.film().develop()

    def render_twostates(self, scene, sensor=0, seed=0, spp=1):
        # Make sure the scene has at least one twostate bsdf
        self.check_scene(scene)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aovs()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos, _ = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L_old, L_new, valid = self.sample_twostates(
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                active=mi.Bool(True)
            )

            # Accumulate into the image block
            alpha = dr.select(valid, mi.Float(1), mi.Float(0))

            dr.schedule(L_old, L_new, alpha)

            self.primal_image_old = self.develop(sensor, ray, weight, pos, spp, L_old, alpha)
            # HACK: Reset the film
            sensor.film().prepare([])
            self.primal_image_new = self.develop(sensor, ray, weight, pos, spp, L_new, alpha)
            # self.primal_image_old = self.primal_image_new

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L_old, L_new, valid, alpha
            gc.collect()

            return self.primal_image_old, self.primal_image_new

    def sample_twostates(self,
                         scene: mi.Scene,
                         sampler: mi.Sampler,
                         ray: mi.Ray3f,
                         active: mi.Bool,
                         **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum,
               mi.Bool, mi.Spectrum]:
        """
        See ``ADIntegrator.sample()`` for a description of this interface and
        the role of the various parameters and return values.
        """

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)     # Depth of current vertex
        L_old = mi.Spectrum(0)   # Old radiance accumulator
        L_new = mi.Spectrum(0)   # New radiance accumulator
        β_old = mi.Spectrum(1)   # Old path throughput weight
        β_new = mi.Spectrum(1)   # New path throughput weight
        η = mi.Float(1)          # Index of refraction
        active = mi.Bool(active) # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        # Record the following loop in its entirety
        loop = mi.Loop(name="Path Replay Backpropagation (%s)",
                       state=lambda: (sampler, ray, depth, L_old, L_new, β_old, β_new, η, active,
                                      prev_si, prev_bsdf_pdf, prev_bsdf_delta))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            si = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All,
                                        coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            mis = mis_weight(
                prev_bsdf_pdf,
                scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            )

            L_old += β_old * mis * ds.emitter.eval(si)
            L_new += β_new * mis * ds.emitter.eval(si)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em)
            active_em &= dr.neq(ds.pdf, 0.0)

            # Evaluate BSDF * cos(theta)
            wo = si.to_local(ds.d)
            bsdf_value_em_new, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
            bsdf_value_em_old = bsdf.eval_old(bsdf_ctx, si, wo, active_em)
            mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))

            L_old += β_old * mis_em * bsdf_value_em_old * em_weight
            L_new += β_new * mis_em * bsdf_value_em_new * em_weight

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                   sampler.next_1d(),
                                                   sampler.next_2d(),
                                                   active_next)
            bsdf_value_old = bsdf.eval_old(bsdf_ctx, si, bsdf_sample.wo, active_next)

            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # ---- Update loop variables based on current interaction -----

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            η *= bsdf_sample.eta
            bsdf_weight_old = dr.select(bsdf_sample.pdf > 0, bsdf_value_old / bsdf_sample.pdf, 0)

            # If the BSDF is delta, use the new value to avoid having a 0 throughput
            # Note that this means we can't differentiate w.r.t. delta BSDFs (which we usually don't want to do anyway)
            β_old *= dr.select(prev_bsdf_delta, bsdf_weight, bsdf_weight_old)
            β_new *= bsdf_weight

            # Information about the current vertex needed by the next iteration

            prev_si = dr.detach(si, True)
            prev_bsdf_pdf = bsdf_sample.pdf

            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β_new)
            active_next &= dr.neq(β_max, 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β_old[rr_active] *= dr.rcp(rr_prob)
            β_new[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            depth[si.is_valid()] += 1
            active = active_next

        valid_ray = dr.neq(depth, 0)    # Ray validity flag for alpha blending
        if self.xyz:
            return mi.srgb_to_xyz(L_old), mi.srgb_to_xyz(L_new), valid_ray
        else:
            return L_old, L_new, valid_ray
