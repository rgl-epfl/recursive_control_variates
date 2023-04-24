from re import M
import mitsuba as mi
import drjit as dr

class TwoStateMedium(mi.Medium):

    def __init__(self, props):
        mi.set_variant('cuda_ad_rgb')
        mi.Medium.__init__(self, props)

        self.old = props["old"]
        self.new = props["new"]
        self.incoming = props["incoming"]

    def sample_interaction_twostates(self: mi.Medium,
                                     ray: mi.Ray3f,
                                     sample: float,
                                     channel: int,
                                     active: bool):

        # Initialize basic medium interaction fields
        mei_new, mint, maxt, active = self.new.prepare_interaction_sampling(ray, active)

        desired_tau = -dr.log(1 - sample)
        if self.new.majorant_grid() is not None:
            # --- Spatially-variying majorant (supergrid).
            # 1. Prepare for DDA traversal
            # Adapted from: https://github.com/francisengelmann/fast_voxel_traversal/blob/9664f0bde1943e69dbd1942f95efc31901fbbd42/main.cpp
            dda_t, dda_tmax, dda_tdelta = self.new.prepare_dda_traversal(
                self.new.majorant_grid(), ray, mint, maxt, active)

            # 2. Traverse the medium with DDA until we reach the desired
            # optical depth.
            active_dda = mi.Bool(active)
            reached = mi.Bool(False)
            tau_acc = mi.Float(0.0)
            dda_loop = mi.Loop(name=f"TwoStateMedium::sample_interaction_dda",
                    state=lambda: (active_dda, reached, dda_t, dda_tmax, tau_acc, mei_new))
            while dda_loop(active_dda):
                # Figure out which axis we hit first.
                # `t_next` is the ray's `t` parameter when hitting that axis.
                t_next = dr.min(dda_tmax)
                got_assigned = mi.Bool(False)
                tmax_update = dr.zeros(mi.Vector3f)
                for k in range(3):
                    active_k = dr.eq(dda_tmax[k], t_next)
                    tmax_update[k] = dr.select(~got_assigned & active_k, dda_tdelta[k], 0)
                    got_assigned |= active_k

                # Lookup and accumulate majorant in current cell.
                mei_new.t[active_dda] = 0.5 * (dda_t + t_next)
                mei_new.p[active_dda] = ray(mei_new.t)
                majorant = dr.maximum(self.old.majorant_grid().eval_1(mei_new, active_dda),
                                        self.new.majorant_grid().eval_1(mei_new, active_dda))
                tau_next = tau_acc + majorant * (t_next - dda_t)

                # For rays that will stop within this cell, figure out
                # the precise `t` parameter where `desired_tau` is reached.
                t_precise = dda_t + (desired_tau - tau_acc) / majorant
                reached |= active_dda & (majorant > 0) & (t_precise < maxt) & (tau_next >= desired_tau)
                dda_t[active_dda] = dr.select(reached, t_precise, t_next)

                # Prepare for next iteration
                active_dda &= ~reached & (t_next < maxt)
                dda_tmax[active_dda] = dda_tmax + tmax_update
                tau_acc[active_dda] = tau_next

            # Adopt the stopping location, making sure to convert to the main
            # ray's parametrization.
            sampled_t = dr.select(reached, dda_t, dr.inf)
        else:
            # --- A single majorant for the whole volume.
            majorant_old = self.old.get_majorant(mei_new, active)
            majorant_new = self.new.get_majorant(mei_new, active)
            combined_extinction = dr.maximum(majorant_old, majorant_new)
            m = mi.ad.integrators.prbvolpath.index_spectrum(combined_extinction, channel)

            sampled_t              = mint + (desired_tau / m)

        valid_mi   = active & (sampled_t <= maxt)

        if self.new.majorant_grid() is not None:
            # Otherwise it was already looked up above
            combined_extinction = dr.maximum(self.new.majorant_grid().eval_1(mei_new, valid_mi),
                                                self.old.majorant_grid().eval_1(mei_new, valid_mi))
            # mei.combined_extinction = dr.detach(m_majorant_grid.eval_1(mei, valid_mei))

        mei_new.t      = dr.select(valid_mi, sampled_t, dr.inf)
        mei_new.p      = ray(sampled_t)
        mei_new.medium = mi.MediumPtr(self)
        mei_new.mint   = mint

        sigma_s_old, _, sigma_t_old = self.old.get_scattering_coefficients(mei_new, valid_mi)
        sigma_s_new, _, sigma_t_new = self.new.get_scattering_coefficients(mei_new, valid_mi)
        # Adjust sigma_n to the true majorant
        sigma_n_old = combined_extinction - sigma_t_old
        sigma_n_new = combined_extinction - sigma_t_new

        mei_old = mi.MediumInteraction3f(mei_new)

        mei_new.combined_extinction = combined_extinction
        mei_old.combined_extinction = combined_extinction

        mei_new.sigma_s, mei_new.sigma_n, mei_new.sigma_t = sigma_s_new, sigma_n_new, sigma_t_new
        mei_old.sigma_s, mei_old.sigma_n, mei_old.sigma_t = sigma_s_old, sigma_n_old, sigma_t_old

        return mei_old, mei_new

    def eval_tr_old(self, mei, si, active):
        t = dr.minimum(mei.t, si.t) - mei.mint
        return dr.exp(-t * self.old.get_majorant(mei, active))

    def eval_tr_new(self, mei, si, active):
        t = dr.minimum(mei.t, si.t) - mei.mint
        return dr.exp(-t * self.new.get_majorant(mei, active))

    def eval_tr_and_pdf(self, mei, si, active):
        return self.new.eval_tr_and_pdf(mei, si, active)
        # t = dr.minimum(mei.t, si.t) - mei.mint
        # return dr.exp(-t * self.new.get_majorant(mei, active)), 1.0

    def intersect_aabb(self, ray):
        return self.new.intersect_aabb(ray)

    def get_majorant(self, mei, active):
        # Here we need to be very careful. We need to make sure that the majorant is always the new one in the adjoint pass,
        # otherwise the adjoint pass will not be able to compute the correct adjoint values for the new medium.
        # For the primal, pass, it should be the maximum of the two majorants.
        # return dr.maximum(self.old.get_majorant(mei, active), self.new.get_majorant(mei, active))
        return self.new.get_majorant(mei, active)

    def get_scattering_coefficients(self, mei, active):
        return self.new.get_scattering_coefficients(mei, active)

    def sample_interaction(self, ray, sample, channel, active):
        return self.new.sample_interaction(ray, sample, channel, active)

    def sample_interaction_real(self, ray, sampler, channel, active):
        return self.new.sample_interaction_real(ray, sampler, channel, active)

    def sample_interaction_drt(self, ray, sampler, channel, active):
        return self.new.sample_interaction_drt(ray, sampler, channel, active)

    def sample_interaction_drrt(self, ray, sampler, channel, active):
        return self.new.sample_interaction_drrt(ray, sampler, channel, active)

    def prepare_interaction_sampling(self, ray, active):
        return self.new.prepare_interaction_sampling(ray, active)

    def prepare_dda_traversal(self, majorant_grid, ray, mint, maxt, active = True):
        return self.new.prepare_dda_traversal(majorant_grid, ray, mint, maxt, active)

    def phase_function(self):
        return self.new.phase_function()

    def old_phase_function(self):
        return self.old.phase_function()

    def use_emitter_sampling(self):
        return self.new.use_emitter_sampling()

    def has_spectral_extinction(self):
        return self.new.has_spectral_extinction()

    def is_homogeneous(self):
        return self.new.is_homogeneous()

    def majorant_grid(self):
        return self.new.majorant_grid()

    def majorant_resolution_factor(self):
        return self.new.majorant_resolution_factor()

    def set_majorant_resolution_factor(self, factor):
        self.old.set_majorant_resolution_factor(factor)
        self.old.parameters_changed()
        self.new.set_majorant_resolution_factor(factor)
        self.new.parameters_changed()

    def has_majorant_grid(self):
        return self.new.has_majorant_grid()

    def majorant_resolution_factor(self):
        return self.new.majorant_resolution_factor()

    def traverse(self, callback):
        self.incoming.traverse(callback)

    def parameters_changed(self, keys):
        old_params = mi.traverse(self.old)
        new_params = mi.traverse(self.new)
        incoming_params = mi.traverse(self.incoming)

        # Hardcoded for diffuse BSDFs
        for key in incoming_params.keys():
            old_params[key] = type(incoming_params[key])(new_params[key])
            new_params[key] = type(incoming_params[key])(incoming_params[key])

        old_params.update()
        new_params.update()

    def get_albedo(self, mei, active):
        return self.new.get_albedo(mei, active)

    def to_string(self):
        old_params = mi.traverse(self.old)
        new_params = mi.traverse(self.new)
        incoming_params = mi.traverse(self.incoming)
        keys = incoming_params.keys()
        # For debugging purposes
        return ('TwoStateMedium[\n'
                '    old_indices=%s,\n'
                '    old_indices_ad=%s,\n'
                '    new_indices=%s,\n'
                '    new_indices_ad=%s,\n'
                '    incoming_indices=%s,\n'
                '    incoming_indices_ad=%s,\n'
                ']' % (
                    [old_params[key].index for key in keys if hasattr(incoming_params[key], 'index')],
                    [old_params[key].index_ad for key in keys if hasattr(incoming_params[key], 'index_ad')],
                    [new_params[key].index for key in keys if hasattr(incoming_params[key], 'index')],
                    [new_params[key].index_ad for key in keys if hasattr(incoming_params[key], 'index_ad')],
                    [incoming_params[key].index for key in keys if hasattr(incoming_params[key], 'index')],
                    [incoming_params[key].index_ad for key in keys if hasattr(incoming_params[key], 'index_ad')]
                ))
