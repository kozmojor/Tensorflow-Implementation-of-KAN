import tensorflow as tf
import numpy as np
from keras.layers import Layer
from .spline import coef2curve, extend_grid, curve2coef
from .utils import sparse_mask
from typing import Tuple, List, Any, Union, Callable

class KANLayer(Layer):
    ''' 
        Initializes the KANLayer with input/output dimensions, grid, scaling factors, and coefficients.
        Args:
            in_dim: int, input dimension.
            out_dim: int, output dimension.
            num: int, number of intervals in the grid.
            k: int, spline order (e.g., cubic spline k=3).
            noise_scale: float, scale for initialization noise.
            scale_base_mu: float, mean for scaling base initialization.
            scale_base_sigma: float, standard deviation for scaling base initialization.
            scale_sp: float, scaling factor for spline values.
            base_fun: callable, base activation function (default: silu).
            grid_eps: float, epsilon for adaptive grid updating.
            grid_range: list, range of grid values.
            sp_trainable: bool, if spline scaling is trainable.
            sb_trainable: bool, if base scaling is trainable.
            device: str, device placement ('cpu' or 'gpu').
            sparse_init: bool, if sparse initialization is used.
        '''

    def __init__(
        self, 
        in_dim=3, 
        out_dim=2, 
        num=5, 
        k=3, 
        noise_scale=0.5, 
        scale_base_mu=0.0, 
        scale_base_sigma=1.0, 
        scale_sp=1.0, 
        base_fun=tf.nn.silu, 
        grid_eps=0.02, 
        grid_range=[-1, 1], 
        sp_trainable=True, 
        sb_trainable=True, 
        device='cpu', 
        sparse_init=False
    ):
        super(KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.grid_eps = grid_eps
        self.base_fun = base_fun

        # Initialize grid
        grid = tf.linspace(grid_range[0], grid_range[1], num + 1)
        grid = tf.tile(grid[None, :], [self.in_dim, 1])
        grid = extend_grid(grid, k_extend=k)
        self.grid = tf.Variable(grid, trainable=False, dtype=tf.float32)

        #Initialize coefficients
        noises = (tf.random.uniform((num + 1, in_dim, out_dim), -0.5, 0.5) * noise_scale) / num
        transposed_grid = tf.transpose(self.grid[:, k:-k], perm=[1, 0])  # 交换维度
        self.coef = tf.Variable(
            curve2coef(transposed_grid, noises, self.grid, k),
            trainable=True,
            dtype=tf.float32
        )
        # self.coef = self.add_weight(
        #     name="spline_kernel",
        #     shape=(self.in_size, self.spline_basis_size, out_dim),
        #     initializer=tf.keras.initializers.RandomNormal(stddev=self.spline_initialize_stddev),
        #     trainable=True,
        #     dtype=self.dtype
        # )

        # Initialize masks
        if sparse_init:
            self.mask = tf.Variable(sparse_mask(in_dim, out_dim), trainable=False, dtype=tf.float32)
        else:
            self.mask = tf.Variable(tf.ones((in_dim, out_dim)), trainable=False, dtype=tf.float32)

        # Initialize scaling factors
        self.scale_base = tf.Variable(
            scale_base_mu * 1 / np.sqrt(in_dim) +
            scale_base_sigma * (tf.random.uniform((in_dim, out_dim), -1, 1)) * 1 / np.sqrt(in_dim),
            trainable=sb_trainable,
            dtype=tf.float32
        )
        self.scale_sp = tf.Variable(
            tf.ones((in_dim, out_dim)) * scale_sp * 1 / np.sqrt(in_dim) * self.mask,
            trainable=sp_trainable,
            dtype=tf.float32
        )

    def call(self, x):
        """
        Forward pass for the KANLayer.

        Args:
            x: Input tensor of shape (batch, in_dim)

        Returns:
            y: Output tensor of shape (batch, out_dim)
        """
        # Pre-activations
        preacts = tf.tile(x[:, None, :], [1, self.out_dim, 1])

        # Residual function (base)
        base = self.base_fun(x)

        # Evaluate splines
        spline_vals = coef2curve(x, self.grid, self.coef, self.k)

        # Combine residual and spline components
        y = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * spline_vals
        y = self.mask[None, :, :] * y

        # Sum over input dimension
        y = tf.reduce_sum(y, axis=1)

        return y

    def update_grid_from_samples(self, x, mode='sample'):
        """
        Update the grid based on input samples.

        Args:
            x: Input tensor of shape (batch, in_dim)
        """
        x_sorted = tf.sort(x, axis=0)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            ids = tf.cast(tf.linspace(0, tf.shape(x_sorted)[0] - 1, num_interval + 1), tf.int32)
            grid_adaptive = tf.gather(x_sorted, ids, axis=0)
            margin = 0.0
            h = (grid_adaptive[:, -1:] - grid_adaptive[:, :1] + 2 * margin) / num_interval
            grid_uniform = grid_adaptive[:, :1] - margin + h * tf.range(num_interval + 1, dtype=tf.float32)
            return self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        grid = get_grid(num_interval)

        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_sorted = tf.transpose(sample_grid)
            y_eval = coef2curve(x_sorted, self.grid, self.coef, self.k)

        self.grid.assign(extend_grid(grid, k_extend=self.k))
        self.coef.assign(curve2coef(tf.transpose(x_sorted), y_eval, self.grid, self.k))