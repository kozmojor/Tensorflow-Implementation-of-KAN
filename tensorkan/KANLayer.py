import tensorflow as tf
import numpy as np
from .spline import extend_grid, coef2curve, curve2coef
from .utils import sparse_mask


class KANLayer(tf.keras.layers.Layer):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.5,
                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
                 base_fun=tf.nn.silu, grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, custom_dtype=tf.float32, sparse_init=False, **kwargs):
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
        grid = tf.cast(grid,tf.float32)
        grid = extend_grid(grid, k_extend=k)
        self.grid = tf.Variable(grid, trainable=False, name="grid")
        # print(self.grid)
        # Initialize coefficients
        noises = (tf.random.uniform((self.num + 1, self.in_dim, self.out_dim), minval=-0.5, maxval=0.5)
                  * noise_scale / num)
        self.coef = tf.Variable(
                                curve2coef(tf.transpose(self.grid[:, k:-k], perm=[1, 0]), noises, self.grid, k),
                                trainable=True,
                                name="coef"
                                )
        
        if sparse_init:
            self.mask = self.add_weight(
                name="mask",
                shape=(self.in_dim, self.out_dim),
                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0),  # 根据需要调整初始化方法
                trainable=False,
                dtype=self.custom_dtype
            )
        else:
            self.mask = self.add_weight(
                name="mask",
                shape=(self.in_dim, self.out_dim),
                initializer=tf.keras.initializers.Ones(),
                trainable=False,
                dtype=self.custom_dtype
            )

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
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "num": self.num,
            "k": self.k,
            "noise_scale": self.noise_scale,
            "scale_base_mu": 0.0,  
            "scale_base_sigma": 1.0,  
            "scale_sp": 1.0,  
            "base_fun": self.base_fun,  
            "grid_eps": self.grid_eps,
            "grid_range": [-1, 1],  
            "sp_trainable": True,  
            "sb_trainable": True,  
            "custom_dtype": str(self.custom_dtype.name), 
            "sparse_init": False  
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        config["custom_dtype"] = tf.dtypes.as_dtype(config["custom_dtype"])  
        config["base_fun"] = tf.keras.activations.get(config["base_fun"])  
        return cls(**config)

    def call(self, inputs):
        """
        KANLayer forward given input x
        
        Args:
        -----
            x : 2D tf.Tensor
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            y : 2D tf.Tensor
                outputs, shape (number of samples, output dimension)
            preacts : 3D tf.Tensor
                fan out x into activations, shape (number of samples, output dimension, input dimension)
            postacts : 3D tf.Tensor
                the outputs of activation functions with preacts as inputs
            postspline : 3D tf.Tensor
                the outputs of spline functions with preacts as inputs
        
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> x = tf.random.normal((100, 3))
        >>> y, preacts, postacts, postspline = model(x)
        >>> y.shape, preacts.shape, postacts.shape, postspline.shape
        """
        inputs = tf.cast(inputs, self.custom_dtype)
        # Pre-activation: broadcasting input to (batch_size, out_dim, in_dim)
        preacts = tf.expand_dims(inputs, axis=1)  # Shape: (batch_size, 1, in_dim)
        preacts = tf.tile(preacts, [1, self.out_dim, 1])  # Shape: (batch_size, out_dim, in_dim)
        # Residual function
        base = self.base_fun(inputs)  # Shape: (batch_size, in_dim)

        # Spline computation
        spline_output = coef2curve(inputs, self.grid, self.coef, self.k)  # Shape: (batch_size, in_dim, out_dim)
        postspline = tf.transpose(spline_output, perm=[0, 2, 1])  # Shape: (batch_size, out_dim, in_dim)

        # Combine residual and spline outputs
        combined = self.scale_base[None, :, :] * base[:, :, None] + self.scale_sp[None, :, :] * spline_output
        combined = self.mask[None, :, :] * combined

        # Post-activation
        postacts = tf.transpose(combined, perm=[0, 2, 1])  # Shape: (batch_size, out_dim, in_dim)

        # Final output
        outputs = tf.reduce_sum(combined, axis=1)  # Sum over input dimension, shape: (batch_size, out_dim)
        return outputs, preacts, postacts, postspline

    def update_grid_from_samples(self, inputs, mode='sample'):
        '''
        Update grid from samples
        
        Args:
        -----
            x : 2D tf.Tensor
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.numpy())
        >>> x = tf.linspace(-3.0, 3.0, 100)[:, tf.newaxis]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.numpy())
        '''
        inputs = tf.cast(inputs, self.custom_dtype)

        batch_size = tf.shape(inputs)[0]
        sorted_inputs = tf.sort(inputs, axis=0)
        spline_output = coef2curve(sorted_inputs, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k

        def compute_grid(num_intervals):
            # ids = tf.concat([tf.cast(batch_size / num_intervals * tf.range(num_intervals), tf.int32), [-1]], axis=0)
            
            last_index = tf.shape(sorted_inputs)[0] - 1
            ids = tf.concat(
                [tf.cast(tf.cast(batch_size, tf.float32) / tf.cast(num_intervals, tf.float32) * tf.cast(
                    tf.range(num_intervals), tf.float32), tf.int32), [last_index]],
                axis=0
            )

            grid_adaptive = tf.transpose(tf.gather(sorted_inputs, ids, axis=0))
            h = (grid_adaptive[:, -1:] - grid_adaptive[:, :1]) / num_intervals
            grid_uniform = grid_adaptive[:, :1] + h * tf.range(num_intervals + 1, dtype=tf.float32)[None, :]
            return self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive

        new_grid = compute_grid(num_interval)
        if mode == 'grid':
            sampled_grid = compute_grid(num_interval)
            sorted_inputs = tf.transpose(sampled_grid)
            spline_output = coef2curve(sorted_inputs, self.grid, self.coef, self.k)

        self.grid.assign(extend_grid(new_grid, k_extend=self.k))
        self.coef.assign(curve2coef(sorted_inputs, spline_output, self.grid, self.k))

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        Update grid from a parent KANLayer and samples.

        Args:
        -----
            parent : KANLayer
                A parent KANLayer (whose grid is usually coarser than the current model).
            x : 2D tf.Tensor
                Inputs, shape (number of samples, input dimension).
            mode : str, optional
                Mode of grid initialization. Default is 'sample'.

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = tf.random.normal((batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid)
        '''
        batch = tf.shape(x)[0]

        # Sort the input tensor along the first axis
        x_pos = tf.sort(x, axis=0)
        last_index = tf.shape(x_pos)[0] - 1

        # Evaluate parent grid coefficients at sorted input positions
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = tf.shape(self.grid)[1] - 1 - 2 * self.k

        def get_grid(num_interval):
            # Calculate adaptive grid based on sorted input
            # print((batch / num_interval).dtype, tf.cast(tf.range(num_interval), tf.float32).dtype)
            ids = tf.concat(
                [tf.cast(tf.cast(batch / num_interval, tf.float32) * tf.cast(tf.range(num_interval), tf.float32), tf.int32), [last_index]], axis=0
            )
            # print(batch,num_interval,tf.range(num_interval),ids)
            grid_adaptive = tf.transpose(tf.gather(x_pos, ids, axis=0), perm=[1, 0])

            # Calculate uniform grid
            h = (grid_adaptive[:, -1:] - grid_adaptive[:, 0:1]) / tf.cast(num_interval, tf.float32)
            grid_uniform = grid_adaptive[:, 0:1] + h * tf.range(num_interval + 1, dtype=tf.float32)[None, :]

            # Weighted combination of adaptive and uniform grid
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid

        grid = get_grid(num_interval)

        # If mode is 'grid', adjust sampling strategy
        if mode == 'grid':
            sample_grid = get_grid(2 * num_interval)
            x_pos = tf.transpose(sample_grid, perm=[1, 0])
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)

        # Extend the grid with additional intervals based on k
        grid = extend_grid(grid, k_extend=self.k)

        # Update the current model's grid and coefficients
        self.grid.assign(grid)
        self.coef.assign(curve2coef(x_pos, y_eval, self.grid, self.k))
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
