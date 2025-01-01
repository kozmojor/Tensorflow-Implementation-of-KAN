import tensorflow as tf
import numpy as np
from .spline import extend_grid, coef2curve, curve2coef
from .utils import sparse_mask


class KANLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5,
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                 base_fun=tf.nn.silu, grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, custom_dtype=tf.float64, sparse_init=False, **kwargs):
        """
        TensorFlow implementation of KANLayer.
        """
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.noise_scale = noise_scale
        self.grid_eps = grid_eps
        self.base_fun = base_fun
        self.custom_dtype = custom_dtype

        # Initialize grid
        grid = tf.linspace(grid_range[0], grid_range[1], num=num + 1)
        grid = tf.tile(tf.expand_dims(grid, 0), [self.in_dim, 1])
        grid = extend_grid(grid, k_extend=k)
        self.grid = tf.Variable(grid, trainable=False, name="grid")

        # Initialize coefficients
        noises = (tf.random.uniform((self.num + 1, self.in_dim, self.out_dim), minval=-0.5, maxval=0.5)
                  * noise_scale / num)
        self.coef = tf.Variable(
                                curve2coef(tf.transpose(self.grid[:, k:-k], perm=[1, 0]), noises, self.grid, k),
                                trainable=True,
                                name="coef"
                                )
        self.coef = self.add_weight(
                    name="coef",
                    shape=(self.in_dim, self.out_dim, self.num + self.k),
                    initializer=tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5),
                    trainable=True,
                    dtype=self.custom_dtype
                    )
        
        # Initialize masks
        if sparse_init:
            self.mask = tf.cast(tf.Variable(sparse_mask(in_dim, out_dim), trainable=False, name="mask"), self.custom_dtype)
        else:
            self.mask = tf.cast(tf.Variable(sparse_mask(in_dim, out_dim), trainable=False, name="mask"), self.custom_dtype)

        # Scale initialization
        self.scale_base = tf.cast(tf.Variable(
            scale_base_mu / np.sqrt(in_dim) +
            scale_base_sigma * tf.random.uniform((in_dim, out_dim), minval=-1, maxval=1) / np.sqrt(in_dim),
            trainable=sb_trainable, name="scale_base"
        ), self.custom_dtype)

        self.scale_base = self.add_weight(
                    name="scale_base",
                shape=(self.in_dim, self.out_dim),
                initializer=tf.keras.initializers.RandomNormal(mean=scale_base_mu, stddev=scale_base_sigma / np.sqrt(self.in_dim)),
                trainable=sb_trainable,
                dtype=self.custom_dtype
                )
        
        self.scale_sp = tf.cast(tf.Variable(
            tf.ones((in_dim, out_dim), dtype=self.custom_dtype) * scale_sp * self.mask,
            trainable=sp_trainable, name="scale_sp"
        ), self.custom_dtype)

        self.scale_sp = self.add_weight(
                name="scale_sp",
                shape=(self.in_dim, self.out_dim),
                initializer=tf.keras.initializers.Ones(),
                trainable=sp_trainable,
                dtype=self.custom_dtype
                )

    def call(self, inputs):
        """
        Forward pass through the KANLayer.
        """
        inputs = tf.cast(inputs, self.custom_dtype)

        batch_size = tf.shape(inputs)[0]

        # Residual function
        base = self.base_fun(inputs)  # Shape: (batch_size, in_dim)

        # Spline computation
        spline_output = coef2curve(inputs, self.grid, self.coef, self.k)  # Shape: (batch_size, in_dim, out_dim)

        # Combine residual and spline outputs
        combined = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * spline_output
        combined = self.mask[None, :, :] * combined

        # Final output
        outputs = tf.reduce_sum(combined, axis=1)  # Sum over input dimension, shape: (batch_size, out_dim)

        return outputs

    def update_grid_from_samples(self, inputs, mode='sample'):
        """
        Update grid using the input samples.
        """
        inputs = tf.cast(inputs, self.custom_dtype)

        batch_size = tf.shape(inputs)[0]
        sorted_inputs = tf.sort(inputs, axis=0)
        spline_output = coef2curve(sorted_inputs, self.grid, self.coef, self.k)

        def compute_grid(num_intervals):
            ids = tf.concat([tf.cast(batch_size / num_intervals * tf.range(num_intervals), tf.int32), [-1]], axis=0)
            grid_adaptive = tf.transpose(tf.gather(sorted_inputs, ids, axis=0))
            h = (grid_adaptive[:, -1:] - grid_adaptive[:, :1]) / num_intervals
            grid_uniform = grid_adaptive[:, :1] + h * tf.range(num_intervals + 1, dtype=tf.float32)[None, :]
            return self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        new_grid = compute_grid(self.num)
        if mode == 'grid':
            sampled_grid = compute_grid(2 * self.num)
            sorted_inputs = tf.transpose(sampled_grid)
            spline_output = coef2curve(sorted_inputs, self.grid, self.coef, self.k)

        self.grid.assign(extend_grid(new_grid, k_extend=self.k))
        self.coef.assign(curve2coef(sorted_inputs, spline_output, self.grid, self.k))

    def get_subset(self, in_ids, out_ids):
        """
        Create a smaller KANLayer by selecting specific inputs and outputs.
        """
        sub_layer = KANLayer(len(in_ids), len(out_ids), self.num, self.k, base_fun=self.base_fun)
        sub_layer.grid.assign(tf.gather(self.grid, in_ids, axis=0))
        sub_layer.coef.assign(tf.gather(tf.gather(self.coef, in_ids, axis=0), out_ids, axis=1))
        sub_layer.scale_base.assign(tf.gather(tf.gather(self.scale_base, in_ids, axis=0), out_ids, axis=1))
        sub_layer.scale_sp.assign(tf.gather(tf.gather(self.scale_sp, in_ids, axis=0), out_ids, axis=1))
        sub_layer.mask.assign(tf.gather(tf.gather(self.mask, in_ids, axis=0), out_ids, axis=1))
        return sub_layer
