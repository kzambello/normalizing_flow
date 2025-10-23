import numpy as np

import os
import re

import tensorflow as tf
import tensorflow_probability as tfp

# Use keras 2
import tf_keras as tfk

os.environ["TF_USE_LEGACY_KERAS"] = "1"

## Use keras 3
# tfk = tf.keras

tfb = tfp.bijectors
tfd = tfp.distributions


class MAF(tfk.Model):
    def __init__(
        self,
        nvars,
        ncondvars,
        nblocks=5,
        hidden_units=[1024, 1024],
        activation="relu",
        last_activation="relu",
        l1=0.0,
        l2=1.0e-6,
    ):
        super().__init__()

        self.nvars = nvars
        self.ncondvars = ncondvars

        self.normalizing_flow = self.build_normalizing_flow(
            nvars, ncondvars, nblocks, hidden_units, activation, last_activation, l1, l2
        )

        # without this, tfk.Model() won't recognize the trainable variables of the flow
        self.call([np.zeros(self.nvars), np.zeros(self.ncondvars)])

    def build_normalizing_flow(
        self,
        nvars,
        ncondvars,
        nblocks,
        hidden_units,
        activation,
        last_activation,
        l1,
        l2,
    ):

        kernel_initializer = tfk.initializers.RandomNormal(
            mean=0.0, stddev=1.0e-3, seed=42
        )

        kernel_regularizer = tfk.regularizers.L1L2(l1=l1, l2=l2)

        bias_regularizer = tfk.regularizers.L1L2(l1=l1, l2=l2)

        distribution = tfd.Sample(tfd.Normal(loc=0.0, scale=1.0), sample_shape=[nvars])

        def build_made():
            made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=hidden_units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                conditional=True,
                event_shape=(nvars,),
                conditional_event_shape=(ncondvars,),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            return made

        def build_made_lastlayer():
            made = tfb.AutoregressiveNetwork(
                params=2,
                hidden_units=hidden_units,
                activation=last_activation,
                kernel_initializer=kernel_initializer,
                conditional=True,
                event_shape=(nvars,),
                conditional_event_shape=(ncondvars,),
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            return made

        bijectors = []
        # reproducible permutations
        permutations = [
            np.random.RandomState(seed=(42 + b)).permutation(np.arange(nvars))
            for b in range(nblocks)
        ]

        for i in range(nblocks):
            if i < (nblocks - 1):
                maf = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=build_made(), name="maf" + str(i)
                )
            else:
                maf = tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=build_made_lastlayer(), name="maf" + str(i)
                )
            bijectors.append(maf)
            bijectors.append(tfb.Permute(permutation=permutations[i]))
            # bijectors.append(tfb.BatchNormalization())

        maf_chain = tfb.Chain(bijectors)

        normalizing_flow = tfd.TransformedDistribution(
            distribution=distribution, bijector=maf_chain
        )

        return normalizing_flow

    def call(self, x):
        return self.normalizing_flow.log_prob(
            x[0],
            bijector_kwargs=self.make_bijector_kwargs(
                self.normalizing_flow.bijector, {"maf.": {"conditional_input": x[1]}}
            ),
        )

    def sample(self, cond, nsamples):

        conditional_input = cond * np.ones((nsamples, self.ncondvars)).reshape(
            -1, self.ncondvars
        )

        barg = self.make_bijector_kwargs(
            self.normalizing_flow.bijector,
            {"maf.": {"conditional_input": conditional_input}},
        )

        samples = self.normalizing_flow.sample(nsamples, bijector_kwargs=barg).numpy()

        log_probs = -self.normalizing_flow.log_prob(
            samples, bijector_kwargs=barg
        ).numpy()

        idx = np.isnan(log_probs) == False  # remove samples with NaN
        samples = samples[idx]
        log_probs = log_probs[idx]

        idx = np.argsort(log_probs)  # sort by (-)log_prob
        samples = samples[idx]
        log_probs = log_probs[idx]

        # print(f"(-)log_prob: {log_probs}")

        return samples

    def make_bijector_kwargs(self, bijector, name_to_kwargs):
        if hasattr(bijector, "bijectors"):
            return {
                b.name: self.make_bijector_kwargs(b, name_to_kwargs)
                for b in bijector.bijectors
            }
        else:
            for name_regex, kwargs in name_to_kwargs.items():
                if re.match(name_regex, bijector.name):
                    return kwargs
        return {}
