import mitsuba as mi
import drjit as dr

class TwoStateBSDF(mi.BSDF):
    # Evaluates two BSDFs with its `eval_2` method

    def __init__(self, props: mi.Properties):
        mi.set_variant('cuda_ad_rgb')
        mi.BSDF.__init__(self, props)

        self.old = props['old']
        self.new = props['new']
        self.incoming = props['incoming']

        # props.set_plugin_name(props['bsdf_name'])
        # del props['bsdf_name']
        # pmgr = mi.PluginManager.instance()
        # self.old = pmgr.create_object(props, pmgr.get_plugin_class(props.plugin_name(), mi.variant()))
        # self.new = pmgr.create_object(props, pmgr.get_plugin_class(props.plugin_name(), mi.variant()))
        # self.incoming = pmgr.create_object(props, pmgr.get_plugin_class(props.plugin_name(), mi.variant()))

        self.m_components  = self.old.m_components
        self.m_flags = self.old.m_flags

    def sample(self, ctx, si, sample1, sample2, active):
        return self.new.sample(ctx, si, sample1, sample2, active)

    def eval(self, ctx, si, wo, active):
        return self.new.eval(ctx, si, wo, active)

    def eval_old(self, ctx, si, wo, active):
        return self.old.eval(ctx, si, wo, active)

    def pdf(self, ctx, si, wo, active):
        return self.new.pdf(ctx, si, wo, active)

    def eval_pdf(self, ctx, si, wo, active):
        return self.new.eval_pdf(ctx, si, wo, active)

    def eval_diffuse_reflectance(self, si, active):
        return self.new.eval_diffuse_reflectance(si, active)

    def traverse(self, callback):
        self.incoming.traverse(callback)

    def parameters_changed(self, keys):
        old_params = mi.traverse(self.old)
        new_params = mi.traverse(self.new)
        incoming_params = mi.traverse(self.incoming)

        for key in incoming_params.keys():
            old_params[key] = type(incoming_params[key])(new_params[key])
            new_params[key] = type(incoming_params[key])(incoming_params[key])

        old_params.update()
        new_params.update()

    def to_string(self):
        old_params = mi.traverse(self.old)
        new_params = mi.traverse(self.new)
        incoming_params = mi.traverse(self.incoming)
        keys = incoming_params.keys()

        # For debugging purposes
        return ('Evolving[\n'
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

mi.register_bsdf('twostate', TwoStateBSDF)
