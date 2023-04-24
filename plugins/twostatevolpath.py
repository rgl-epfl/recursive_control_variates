from __future__ import annotations as __annotations__ # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr
import gc
from .twostatemedium import TwoStateMedium
from .twostatebsdf import TwoStateBSDF

mis_weight = mi.ad.common.mis_weight
index_spectrum = mi.ad.integrators.prbvolpath.index_spectrum

class TwoStatePRBVolpathIntegrator(mi.ad.integrators.prbvolpath.PRBVolpathIntegrator):
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
            for medium in [shape.interior_medium(), shape.exterior_medium()]:
                has_twostate = has_twostate or isinstance(medium, TwoStateMedium)

        if not has_twostate:
            raise RuntimeError("No TwoStateMedium found in the scene!")

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
        # Make sure the scene has at least one twostate medium
        self.check_scene(scene)

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Always handle null scattering
        self.prepare_scene(scene) # Make sure the flags are properly set first

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

        ray = mi.Ray3f(ray)
        depth = mi.UInt32(0)                          # Depth of current vertex
        L_old = mi.Spectrum(0)   # Old radiance accumulator
        L_new = mi.Spectrum(0)   # New radiance accumulator
        throughput_old = mi.Spectrum(1)                   # Path throughput weight
        throughput_new = mi.Spectrum(1)                   # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)

        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        last_scatter_event = dr.zeros(mi.Interaction3f)
        last_scatter_direction_pdf = mi.Float(1.0)

        medium = dr.zeros(mi.MediumPtr)

        channel = 0
        depth = mi.UInt32(0)
        valid_ray = mi.Bool(False)
        specular_chain = mi.Bool(True)

        if mi.is_rgb: # Sample a color channel to sample free-flight distances
            n_channels = dr.size_v(mi.Spectrum)
            channel = dr.minimum(n_channels * sampler.next_1d(active), n_channels - 1)

        loop = mi.Loop(name=f"TwoState Path Replay Backpropagation",
                    state=lambda: (sampler, active, depth, ray, medium, si,
                                   throughput_old, throughput_new, L_old, L_new, needs_intersection,
                                   last_scatter_event, specular_chain, η,
                                   last_scatter_direction_pdf, valid_ray))
        while loop(active):
            # Russian Roulette
            active &= (dr.any(dr.neq(throughput_new, 0.0)) | dr.any(dr.neq(throughput_old, 0.0)))
            q = dr.minimum(dr.max(dr.maximum(throughput_new, throughput_old)) * dr.sqr(η), 0.99)
            perform_rr = (depth > self.rr_depth)
            active &= (sampler.next_1d(active) < q) | ~perform_rr

            throughput_new[perform_rr] = throughput_new * dr.rcp(q)
            throughput_old[perform_rr] = throughput_old * dr.rcp(q)

            active_medium = active & dr.neq(medium, None) # TODO this is not necessary
            active_surface = active & ~active_medium

            # Handle medium sampling and potential medium escape
            u = sampler.next_1d(active_medium)
            # Sample an interaction according to the maximum of the old&new majorants
            mei_old, mei_new = medium.sample_interaction_twostates(ray, u, channel, active_medium)
            mei_new.t = dr.detach(mei_new.t)

            ray.maxt[active_medium & medium.is_homogeneous() & mei_new.is_valid()] = mei_new.t
            intersect = needs_intersection & active_medium
            si_new = scene.ray_intersect(ray, intersect)
            si[intersect] = si_new

            needs_intersection &= ~active_medium
            mei_new.t[active_medium & (si.t < mei_new.t)] = dr.inf
            mei_old.t[active_medium & (si.t < mei_old.t)] = dr.inf

            # Evaluate transmittance. Is only used for homogeneous media
            if not self.handle_null_scattering:
                tr_new, free_flight_pdf = medium.eval_tr_and_pdf(mei_new, si, active_medium)
                tr_pdf = index_spectrum(free_flight_pdf, channel)

            weight_new = mi.Spectrum(1.0)
            weight_old = mi.Spectrum(1.0)

            escaped_medium = active_medium & ~mei_new.is_valid()
            active_medium &= mei_new.is_valid()
            # Handle null and real scatter events
            if self.handle_null_scattering:
                # Scattering probability is the average of the two states, since we need to be sure we sample null interactions
                # with non-zero probability to ensure unbiasedness
                majorant = index_spectrum(mei_new.combined_extinction, channel) # Both medium interactions have the same majorant
                scatter_prob_new = index_spectrum(mei_new.sigma_t, channel) / majorant
                scatter_prob_old = index_spectrum(mei_old.sigma_t, channel) / majorant
                scatter_prob = dr.select(dr.neq(majorant, 0.0), (scatter_prob_new + scatter_prob_old) * 0.5, 0.0)

                act_null_scatter = (sampler.next_1d(active_medium) >= scatter_prob) & active_medium
                act_medium_scatter = ~act_null_scatter & active_medium

                weight_new[act_null_scatter] *= 2 * mei_new.sigma_n / (mei_new.sigma_n + mei_old.sigma_n)
                weight_old[act_null_scatter] *= 2 * mei_old.sigma_n / (mei_new.sigma_n + mei_old.sigma_n)
            else:
                scatter_prob = mi.Float(1.0)
                t = dr.minimum(mei_new.t, si.t) - mei_new.mint
                tr_new_true = dr.exp(-t * mei_new.sigma_t)
                tr_old_true = dr.exp(-t * mei_old.sigma_t)
                ratio_new = dr.select(tr_pdf > 0.0, tr_new_true / dr.detach(tr_pdf), 0.0)
                ratio_old = dr.select(tr_pdf > 0.0, tr_old_true / dr.detach(tr_pdf), 0.0)

                # Rays that exit the medium do not get their throughput modified by the weight, as it usually cancels out
                # Here we need to do it since the pdf is not the usual one
                throughput_new[escaped_medium] *= ratio_new
                throughput_old[escaped_medium] *= ratio_old

                weight_new[active_medium] *= ratio_new
                weight_old[active_medium] *= ratio_old
                act_medium_scatter = active_medium

            depth[act_medium_scatter] += 1
            last_scatter_event[act_medium_scatter] = dr.detach(mei_new)

            # Don't estimate lighting if we exceeded number of bounces
            active &= depth < self.max_depth
            act_medium_scatter &= active
            if self.handle_null_scattering:
                ray.o[act_null_scatter] = dr.detach(mei_new.p)
                si.t[act_null_scatter] = si.t - dr.detach(mei_new.t)

                weight_new[act_medium_scatter] *= 2 * mei_new.sigma_s / (mei_new.sigma_t + mei_old.sigma_t)
                weight_old[act_medium_scatter] *= 2 * mei_old.sigma_s / (mei_new.sigma_t + mei_old.sigma_t)
            else:
                weight_new[act_medium_scatter] *= mei_new.sigma_s
                weight_old[act_medium_scatter] *= mei_old.sigma_s

            throughput_new[active_medium] *= dr.detach(weight_new)
            throughput_old[active_medium] *= dr.detach(weight_old)

            phase_ctx = mi.PhaseFunctionContext(sampler)
            phase_new = mei_new.medium.phase_function()
            phase_old = mei_old.medium.phase_function()
            phase_new[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)
            phase_old[~act_medium_scatter] = dr.zeros(mi.PhaseFunctionPtr)

            valid_ray |= act_medium_scatter
            wo, phase_pdf = phase_new.sample(phase_ctx, mei_new, sampler.next_1d(act_medium_scatter), sampler.next_2d(act_medium_scatter), act_medium_scatter)
            act_medium_scatter &= phase_pdf > 0.0

            new_ray = mei_new.spawn_ray(wo)
            ray[act_medium_scatter] = new_ray
            needs_intersection |= act_medium_scatter
            last_scatter_direction_pdf[act_medium_scatter] = phase_pdf

            #--------------------- Surface Interactions ---------------------
            active_surface |= escaped_medium
            intersect = active_surface & needs_intersection
            si[intersect] = scene.ray_intersect(ray, intersect)

            # ---------------- Intersection with emitters ----------------
            ray_from_camera = active_surface & dr.eq(depth, 0)
            count_direct = ray_from_camera | specular_chain
            emitter = si.emitter(scene)
            active_e = active_surface & dr.neq(emitter, None) & ~(dr.eq(depth, 0) & self.hide_emitters)

            # Get the PDF of sampling this emitter using next event estimation
            ds = mi.DirectionSample3f(scene, si, last_scatter_event)
            if self.use_nee:
                emitter_pdf = scene.pdf_emitter_direction(last_scatter_event, ds, active_e)
            else:
                emitter_pdf = 0.0
            emitted = emitter.eval(si, active_e)

            mis_bsdf = mis_weight(last_scatter_direction_pdf, emitter_pdf)

            L_new[active_e] += dr.select(count_direct, throughput_new * emitted,
                                throughput_new * mis_bsdf * emitted)

            L_old[active_e] += dr.select(count_direct, throughput_old * emitted,
                                throughput_old * mis_bsdf * emitted)

            active_surface &= si.is_valid()
            ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)

            # --------------------- Emitter sampling ---------------------
            if self.use_nee:
                active_e_surface = active_surface & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth) & (depth + 1 < self.max_depth)
                sample_emitters = mei_new.medium.use_emitter_sampling()
                specular_chain &= ~act_medium_scatter
                specular_chain |= act_medium_scatter & ~sample_emitters
                active_e_medium = act_medium_scatter & sample_emitters
                active_e = active_e_surface | active_e_medium
                ref_interaction = dr.zeros(mi.Interaction3f)
                ref_interaction[act_medium_scatter] = mei_new
                ref_interaction[active_surface] = si
                emitted_old, emitted_new, ds = self.sample_emitter_twostates(ref_interaction, scene, sampler, medium, channel, active_e)
                # Query the BSDF for that emitter-sampled direction
                # For surfaces
                wo_em = si.to_local(ds.d)
                bsdf_val_new, bsdf_pdf = bsdf.eval_pdf(ctx, si, wo_em, active_e_surface)
                bsdf_val_old = bsdf.eval_old(ctx, si, wo_em, active_e_surface)

                # For media
                phase_val_new = phase_new.eval(phase_ctx, mei_new, ds.d, active_e_medium)
                phase_val_old = phase_old.eval(phase_ctx, mei_old, ds.d, active_e_medium)

                nee_weight_new = dr.select(active_e_surface, bsdf_val_new, phase_val_new)
                nee_weight_old = dr.select(active_e_surface, bsdf_val_old, phase_val_old)

                nee_directional_pdf = dr.select(ds.delta, 0.0, dr.select(active_e_surface, bsdf_pdf, phase_val_new))

                mis_em = mis_weight(ds.pdf, nee_directional_pdf)

                L_new[active] += throughput_new * nee_weight_new * mis_em * emitted_new
                L_old[active] += throughput_old * nee_weight_old * mis_em * emitted_old

            # ----------------------- BSDF sampling ----------------------
            bs, bsdf_weight = bsdf.sample(ctx, si, sampler.next_1d(active_surface),
                                    sampler.next_2d(active_surface), active_surface)
            active_surface &= bs.pdf > 0

            bsdf_value_old = bsdf.eval_old(ctx, si, bs.wo, active_surface)
            prev_bsdf_delta = mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)

            bsdf_weight_old = dr.select(bs.pdf > 0, bsdf_value_old / bs.pdf, 0)
            throughput_old[active_surface] *= dr.select(prev_bsdf_delta, bsdf_weight, bsdf_weight_old)

            throughput_new[active_surface] *= bsdf_weight

            # Update the old throughput with the phase/pdf ratio, since it does not cancel out perfectly anymore
            phase_val = phase_old.eval(phase_ctx, mei_old, wo, act_medium_scatter)
            throughput_old[act_medium_scatter] *= phase_val / phase_pdf

            η[active_surface] *= bs.eta
            bsdf_ray = si.spawn_ray(si.to_world(bs.wo))
            ray[active_surface] = bsdf_ray

            needs_intersection |= active_surface
            non_null_bsdf = active_surface & ~mi.has_flag(bs.sampled_type, mi.BSDFFlags.Null)
            depth[non_null_bsdf] += 1

            # update the last scatter PDF event if we encountered a non-null scatter event
            last_scatter_event[non_null_bsdf] = si
            last_scatter_direction_pdf[non_null_bsdf] = bs.pdf

            valid_ray |= non_null_bsdf
            specular_chain |= non_null_bsdf & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Delta)
            specular_chain &= ~(active_surface & mi.has_flag(bs.sampled_type, mi.BSDFFlags.Smooth))
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)
            active &= (active_surface | active_medium)

        if self.xyz:
            return mi.srgb_to_xyz(L_old), mi.srgb_to_xyz(L_new), valid_ray
        else:
            return L_old, L_new, valid_ray

    def sample_emitter_twostates(self, ref_interaction, scene, sampler, medium, channel,
                       active):

        active = mi.Bool(active)
        medium = dr.select(active, medium, dr.zeros(mi.MediumPtr))

        ds, emitter_val = scene.sample_emitter_direction(ref_interaction, sampler.next_2d(active), False, active)
        ds = dr.detach(ds)
        invalid = dr.eq(ds.pdf, 0.0)
        emitter_val[invalid] = 0.0
        active &= ~invalid

        ray = ref_interaction.spawn_ray(ds.d)
        total_dist = mi.Float(0.0)
        si = dr.zeros(mi.SurfaceInteraction3f)
        needs_intersection = mi.Bool(True)
        transmittance_old = mi.Spectrum(1.0)
        transmittance_new = mi.Spectrum(1.0)
        loop = mi.Loop(name=f"PRB Next Event Estimation (twostates)",
                       state=lambda: (sampler, active, medium, ray, total_dist,
                                      needs_intersection, si, transmittance_old, transmittance_new))
        while loop(active):
            remaining_dist = ds.dist * (1.0 - mi.math.ShadowEpsilon) - total_dist
            ray.maxt = dr.detach(remaining_dist)
            active &= remaining_dist > 0.0

            # This ray will not intersect if it reached the end of the segment
            needs_intersection &= active
            si[needs_intersection] = scene.ray_intersect(ray, needs_intersection)
            needs_intersection &= False

            active_medium = active & dr.neq(medium, None)
            active_surface = active & ~active_medium

            # Handle medium interactions / transmittance
            mei_old, mei_new = medium.sample_interaction_twostates(ray, sampler.next_1d(active_medium), channel, active_medium)
            mei_old.t[active_medium & (si.t < mei_old.t)] = dr.inf
            mei_new.t[active_medium & (si.t < mei_new.t)] = dr.inf

            tr_multiplier_old = mi.Spectrum(1.0)
            tr_multiplier_new = mi.Spectrum(1.0)

            # Special case for homogeneous media: directly advance to the next surface / end of the segment
            if self.nee_handle_homogeneous:
                active_homogeneous = active_medium & medium.is_homogeneous()
                mei_old.t[active_homogeneous] = dr.minimum(remaining_dist, si.t)
                mei_new.t[active_homogeneous] = dr.minimum(remaining_dist, si.t)

                tr_multiplier_old[active_homogeneous] = medium.eval_tr_old(mei_old, si, active_homogeneous)
                tr_multiplier_new[active_homogeneous] = medium.eval_tr_new(mei_new, si, active_homogeneous)

                mei_old.t[active_homogeneous] = dr.inf
                mei_new.t[active_homogeneous] = dr.inf

            escaped_medium = active_medium & ~mei_new.is_valid()

            # Ratio tracking transmittance computation
            active_medium &= mei_new.is_valid()
            ray.o[active_medium] = dr.detach(mei_new.p)
            si.t[active_medium] = dr.detach(si.t - mei_new.t)
            tr_multiplier_old[active_medium] *= dr.select(dr.neq(mei_old.combined_extinction, 0.0),
                                                            mei_old.sigma_n / mei_old.combined_extinction,
                                                            mei_old.sigma_n)
            tr_multiplier_new[active_medium] *= dr.select(dr.neq(mei_new.combined_extinction, 0.0),
                                                            mei_new.sigma_n / mei_new.combined_extinction,
                                                            mei_new.sigma_n)

            # Handle interactions with surfaces
            active_surface |= escaped_medium
            active_surface &= si.is_valid() & ~active_medium
            bsdf = si.bsdf(ray)
            bsdf_val = bsdf.eval_null_transmission(si, active_surface)
            tr_multiplier_old[active_surface] = tr_multiplier_old * bsdf_val
            tr_multiplier_new[active_surface] = tr_multiplier_new * bsdf_val

            transmittance_old *= dr.detach(tr_multiplier_old)
            transmittance_new *= dr.detach(tr_multiplier_new)

            # Update the ray with new origin & t parameter
            new_ray = si.spawn_ray(mi.Vector3f(ray.d))
            ray[active_surface] = dr.detach(new_ray)
            ray.maxt = dr.detach(remaining_dist)
            needs_intersection |= active_surface

            # Continue tracing through scene if non-zero weights exist
            active &= (active_medium | active_surface) & (dr.any(dr.neq(transmittance_new, 0.0)) | dr.any(dr.neq(transmittance_old, 0.0)))
            total_dist[active] += dr.select(active_medium, mei_new.t, si.t)

            # If a medium transition is taking place: Update the medium pointer
            has_medium_trans = active_surface & si.is_medium_transition()
            medium[has_medium_trans] = si.target_medium(ray.d)

        return emitter_val * transmittance_old, emitter_val * transmittance_new, ds

    def to_string(self):
        return f'TwoStatePRBVolpathIntegrator[max_depth = {self.max_depth}]'
