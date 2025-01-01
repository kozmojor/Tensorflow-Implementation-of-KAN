import tensorflow as tf
import numpy as np
from .KANLayer import KANLayer
# from .Symbolic_MultKANLayer import *
from .Symbolic_KANLayer import Symbolic_KANLayer
from .LBFGS import *
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
# from .MultKANLayer import MultKANLayer
import pandas as pd
from sympy.printing import latex
from sympy import *
import sympy
import yaml
from .spline import curve2coef
from .utils import SYMBOLIC_LIB


# from .hypothesis import plot_tree
# from .LBFGS import *

class MultKAN(tf.keras.Model):
    '''
    KAN class

    Attributes:
    -----------
        grid : int
            the number of grid intervals
        k : int
            spline order
        act_fun : a list of KANLayers
        symbolic_fun: a list of Symbolic_KANLayer
        depth : int
            depth of KAN
        width : list
            number of neurons in each layer.
            Without multiplication nodes, [2,5,5,3] means 2D inputs, 3D outputs, with 2 layers of 5 hidden neurons.
            With multiplication nodes, [2,[5,3],[5,1],3] means besides the [2,5,53] KAN, there are 3 (1) mul nodes in layer 1 (2).
        mult_arity : int, or list of int lists
            multiplication arity for each multiplication node (the number of numbers to be multiplied)
        grid : int
            the number of grid intervals
        k : int
            the order of piecewise polynomial
        base_fun : fun
            residual function b(x). an activation function phi(x) = sb_scale * b(x) + sp_scale * spline(x)
        symbolic_fun : a list of Symbolic_KANLayer
            Symbolic_KANLayers
        symbolic_enabled : bool
            If False, the symbolic front is not computed (to save time). Default: True.
        width_in : list
            The number of input neurons for each layer
        width_out : list
            The number of output neurons for each layer
        base_fun_name : str
            The base function b(x)
        grip_eps : float
            The parameter that interpolates between uniform grid and adaptive grid (based on sample quantile)
        node_bias : a list of 1D torch.float
        node_scale : a list of 1D torch.float
        subnode_bias : a list of 1D torch.float
        subnode_scale : a list of 1D torch.float
        symbolic_enabled : bool
            when symbolic_enabled = False, the symbolic branch (symbolic_fun) will be ignored in computation (set to zero)
        affine_trainable : bool
            indicate whether affine parameters are trainable (node_bias, node_scale, subnode_bias, subnode_scale)
        sp_trainable : bool
            indicate whether the overall magnitude of splines is trainable
        sb_trainable : bool
            indicate whether the overall magnitude of base function is trainable
        save_act : bool
            indicate whether intermediate activations are saved in forward pass
        node_scores : None or list of 1D torch.float
            node attribution score
        edge_scores : None or list of 2D torch.float
            edge attribution score
        subnode_scores : None or list of 1D torch.float
            subnode attribution score
        cache_data : None or 2D torch.float
            cached input data
        acts : None or a list of 2D torch.float
            activations on nodes
        auto_save : bool
            indicate whether to automatically save a checkpoint once the model is modified
        state_id : int
            the state of the model (used to save checkpoint)
        ckpt_path : str
            the folder to store checkpoints
        round : int
            the number of times rewind() has been called
        device : str
    '''

    def __init__(self, width=None, grid=3, k=3, mult_arity=2, noise_scale=0., scale_base_mu=0.0, scale_base_sigma=1.0,
                 base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1],
                 sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True,
                 first_init=True, ckpt_path='./model', state_id=0, round=0):
        '''
        initalize a KAN model

        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True.
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str

        Returns:
        --------
            self

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        checkpoint directory created: ./model
        saving model version 0.0
        '''
        super().__init__()

        tf.random.set_seed(seed)
        np.random.seed(seed)

        ### initializeing the numerical front ###

        self.act_fun = []
        self.depth = len(width) - 1

        for i in range(len(width)):
            if isinstance(width[i], int):
                width[i] = [width[i], 0]

        self.width = width

        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        self.mult_homo = isinstance(mult_arity, int)
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = tf.nn.swish
        elif base_fun == 'identity':
            base_fun = tf.identity
        elif base_fun == 'zero':
            base_fun = lambda x: x * 0

        self.grid_eps = grid_eps
        self.grid_range = grid_range

        # for l in range(self.depth):
        #     # splines
        #     sp_batch = KANLayer(
        #         in_dim=width_in[l], out_dim=width_out[l+1], num=grid, k=k, noise_scale=noise_scale,
        #         scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma, scale_sp=1.0,
        #         base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range,
        #         sp_trainable=sp_trainable, sb_trainable=sb_trainable, sparse_init=sparse_init
        #     )
        #     self.act_fun.append(sp_batch)

        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []

        # for l in range(self.depth):
        #     self.node_bias.append(tf.Variable(tf.zeros([width_in[l+1]]), trainable=affine_trainable))
        #     self.node_scale.append(tf.Variable(tf.ones([width_in[l+1]]), trainable=affine_trainable))
        #     self.subnode_bias.append(tf.Variable(tf.zeros([width_out[l+1]]), trainable=affine_trainable))
        #     self.subnode_scale.append(tf.Variable(tf.ones([width_out[l+1]]), trainable=affine_trainable))

        self.noise_scale = noise_scale
        self.scale_base_mu = scale_base_mu
        self.scale_base_sigma = scale_base_sigma
        self.sparse_init = sparse_init

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        self.symbolic_fun = [
            Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l + 1]) for l in range(self.depth)
        ]

        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable

        self.save_act = save_act
        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None
        self.cache_data = None
        self.acts = None

        self.auto_save = auto_save
        self.state_id = state_id
        self.ckpt_path = ckpt_path
        self.round = round


        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path + '/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path + '/' + '0.0')
            else:
                self.state_id = state_id
        self.input_id = tf.range(self.width_in[0], dtype=tf.int32)

        self.build(input_shape=(None, width[0][0]))



    def build(self, input_shape):
        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []

        input_dim = input_shape[-1]
        for l in range(self.depth):
            self.node_bias.append(
                self.add_weight(
                    shape=(self.width_in[l + 1],),
                    initializer="zeros",
                    trainable=self.affine_trainable,
                    name=f"node_bias_{l}"
                )
            )
            self.node_scale.append(
                self.add_weight(
                    shape=(self.width_in[l + 1],),
                    initializer="ones",
                    trainable=self.affine_trainable,
                    name=f"node_scale_{l}"
                )
            )
            self.subnode_bias.append(
                self.add_weight(
                    shape=(self.width_out[l + 1],),
                    initializer="zeros",
                    trainable=self.affine_trainable,
                    name=f"subnode_bias_{l}"
                )
            )
            self.subnode_scale.append(
                self.add_weight(
                    shape=(self.width_out[l + 1],),
                    initializer="ones",
                    trainable=self.affine_trainable,
                    name=f"subnode_scale_{l}"
                )
            )

            sp_layer = KANLayer(
                in_dim=self.width_in[l],
                out_dim=self.width_out[l + 1],
                num=self.grid,
                k=self.k,
                noise_scale=self.noise_scale,
                scale_base_mu=self.scale_base_mu,
                scale_base_sigma=self.scale_base_sigma,
                scale_sp=1.0,
                base_fun=self.base_fun,
                grid_eps=self.grid_eps,
                grid_range=self.grid_range,
                sp_trainable=self.sp_trainable,
                sb_trainable=self.sb_trainable,
                sparse_init=self.sparse_init
            )
            self.act_fun.append(sp_layer)

        super().build(input_shape)

    @property
    def width_in(self):
        '''
        The number of input nodes for each layer
        '''
        width = self.width
        width_in = [width[l][0] + width[l][1] for l in range(len(width))]
        return width_in

    @property
    def width_out(self):
        '''
        The number of output subnodes for each layer
        '''
        width = self.width
        if self.mult_homo == True:
            width_out = [width[l][0] + self.mult_arity * width[l][1] for l in range(len(width))]
        else:
            width_out = [width[l][0] + int(np.sum(self.mult_arity[l])) for l in range(len(width))]
        return width_out

    @property
    def n_sum(self):
        '''
        The number of addition nodes for each layer
        '''
        width = self.width
        n_sum = [width[l][0] for l in range(1, len(width) - 1)]
        return n_sum

    @property
    def n_mult(self):
        '''
        The number of multiplication nodes for each layer
        '''
        width = self.width
        n_mult = [width[l][1] for l in range(1, len(width) - 1)]
        return n_mult

    @property
    def feature_score(self):
        '''
        attribution scores for inputs
        '''
        # 待确认
        self.attribute()
        if self.node_scores == None:
            return None
        else:
            return self.node_scores[0]

    def initialize_from_another_model(self, another_model, x):
        '''
        initialize from another model of the same width, but their 'grid' parameter can be different.
        Note this is equivalent to refine() when we don't want to keep another_model

        Args:
        -----
            another_model : MultKAN
            x : 2D torch.float

        Returns:
        --------
            self
        '''
        _ = another_model(x)  # get activations

        self.initialize_grid_from_another_model(another_model, x)

        for l in range(self.depth):
            spb = self.act_fun[l]
            # spb_parent = another_model.act_fun[l]
            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            # Convert curve2coef computations
            spb.coef.assign(
                curve2coef(
                    preacts[:, 0, :],
                    tf.transpose(postsplines, perm=[0, 2, 1]),
                    spb.grid,
                    k=spb.k
                )
            )
            spb.scale_base.assign(another_model.act_fun[l].scale_base)
            spb.scale_sp.assign(another_model.act_fun[l].scale_sp)
            spb.mask.assign(another_model.act_fun[l].mask)

        for l in range(self.depth):
            self.node_bias[l].assign(another_model.node_bias[l])
            self.node_scale[l].assign(another_model.node_scale[l])
            self.subnode_bias[l].assign(another_model.subnode_bias[l])
            self.subnode_scale[l].assign(another_model.subnode_scale[l])

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        return self
    def get_config(self):
        config = super().get_config()
        config.update({
            "width": self.width,
            "grid": self.grid,
            "k": self.k,
            "mult_arity": self.mult_arity,
            "noise_scale": self.noise_scale,
            "base_fun": self.base_fun_name,  # 存储 base_fun 名称
            "symbolic_enabled": self.symbolic_enabled,
            "affine_trainable": self.affine_trainable,
            "grid_eps": self.grid_eps,
            "grid_range": self.grid_range,
            "sp_trainable": self.sp_trainable,
            "sb_trainable": self.sb_trainable,
            "seed": 1,
            "save_act": self.save_act
        })
        return config

    @classmethod
    def from_config(cls, config):
        config["base_fun"] = tf.keras.activations.get(config["base_fun"])
        return cls(**config)

    def log_history(self, method_name):

        if self.auto_save:
            history_path = os.path.join(self.ckpt_path, 'history.txt')
            with open(history_path, 'a') as file:
                file.write(f'{self.round}.{self.state_id} => {method_name} => {self.round}.{self.state_id + 1}\n')

            self.state_id += 1

            save_path = os.path.join(self.ckpt_path, f'{self.round}.{self.state_id}')
            self.saveckpt(path=save_path)
            print(f'Saving model version {self.round}.{self.state_id}')

    def refine(self, new_grid):
        '''
        grid refinement

        Args:
        -----
            new_grid : init
                the number of grid intervals after refinement

        Returns:
        --------
            a refined model : MultKAN
        '''

        model_new = MultKAN(width=self.width,
                            grid=new_grid,
                            k=self.k,
                            mult_arity=self.mult_arity,
                            base_fun=self.base_fun_name,
                            symbolic_enabled=self.symbolic_enabled,
                            affine_trainable=self.affine_trainable,
                            grid_eps=self.grid_eps,
                            grid_range=self.grid_range,
                            sp_trainable=self.sp_trainable,
                            sb_trainable=self.sb_trainable,
                            ckpt_path=self.ckpt_path,
                            auto_save=True,
                            first_init=False,
                            state_id=self.state_id,
                            round=self.round)

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid

        self.log_history('refine')
        model_new.state_id += 1

        return model_new

    # def save_weights(self, path):
    #     weights = {}
    #     for i, layer in enumerate(self.act_fun):
    #         weights[f'layer_{i}'] = [w.numpy() for w in layer.trainable_weights]
    #     np.savez(path, **weights)

    # def load_weights(self, path):
    #     data = np.load(path)
    #     for i, layer in enumerate(self.act_fun):
    #         for w, saved_w in zip(layer.trainable_weights, data[f'layer_{i}']):
    #             w.assign(saved_w)

    def saveckpt(self, path='model'):
        '''
        save the current model to files (configuration file and state file)

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            None

        '''

        dic = dict(
            width=self.width,
            grid=self.grid,
            k=self.k,
            mult_arity=self.mult_arity,
            base_fun_name=self.base_fun_name,
            symbolic_enabled=self.symbolic_enabled,
            affine_trainable=self.affine_trainable,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
            sp_trainable=self.sp_trainable,
            sb_trainable=self.sb_trainable,
            state_id=self.state_id,
            auto_save=self.auto_save,
            ckpt_path=self.ckpt_path,
            round=self.round,
        )

        for i in range(self.depth):
            dic[f'symbolic.funs_name.{i}'] = self.symbolic_fun[i].funs_name

        dic = dict()

        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        self.save_weights(f'{path}_state.weights.h5')

        if self.cache_data is not None:
            np.save(f'{path}_cache_data.npy', self.cache_data.numpy())

    @staticmethod
    def loadckpt(path='model'):
        '''
        load checkpoint from path

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MultKAN

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        >>> KAN.loadckpt('./mark')
        '''
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)

        model_load = MultKAN(
            width=config['width'],
            grid=config['grid'],
            k=config['k'],
            mult_arity=config['mult_arity'],
            base_fun=config['base_fun_name'],
            symbolic_enabled=config['symbolic_enabled'],
            affine_trainable=config['affine_trainable'],
            grid_eps=config['grid_eps'],
            grid_range=config['grid_range'],
            sp_trainable=config['sp_trainable'],
            sb_trainable=config['sb_trainable'],
            state_id=config['state_id'],
            auto_save=config['auto_save'],
            first_init=False,
            ckpt_path=config['ckpt_path'],
            round=config['round'] + 1,
        )

        model_load.load_weights(f'{path}_state.weights.h5')

        try:
            cache_data = np.load(f'{path}_cache_data.npy')
            model_load.cache_data = tf.convert_to_tensor(cache_data, dtype=tf.float32)
        except FileNotFoundError:
            model_load.cache_data = None

        depth = len(model_load.width) - 1
        for l in range(depth):
            out_dim = model_load.symbolic_fun[l].out_dim
            in_dim = model_load.symbolic_fun[l].in_dim
            funs_name = config[f'symbolic.funs_name.{l}']
            for j in range(out_dim):
                for i in range(in_dim):
                    fun_name = funs_name[j][i]
                    model_load.symbolic_fun[l].funs_name[j][i] = fun_name
                    model_load.symbolic_fun[l].funs[j][i] = SYMBOLIC_LIB[fun_name][0]
                    model_load.symbolic_fun[l].funs_sympy[j][i] = SYMBOLIC_LIB[fun_name][1]
                    model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = SYMBOLIC_LIB[fun_name][3]

        return model_load

    def copy(self):
        '''
        deepcopy

        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MultKAN
        '''
        path = 'copy_temp'
        self.saveckpt(path)
        return MultKAN.loadckpt(path)

    def rewind(self, model_id):
        '''
        rewind to an old version

        Args:
        -----
            model_id : str
                in format '{a}.{b}' where a is the round number, b is the version number in that round

        Returns:
        --------
            MultKAN

        '''
        self.round += 1
        self.state_id = model_id.split('.')[-1]

        # Update history
        history_path = f'{self.ckpt_path}/history.txt'
        with open(history_path, 'a') as file:
            file.write(f'### Round {self.round} ###\n')

        # Save the new state
        self.saveckpt(path=f'{self.ckpt_path}/{self.round}.{self.state_id}')

        print(f'Rewind to model version {int(self.round) - 1}.{self.state_id}, renamed as {self.round}.{self.state_id}')

        # Return the rewound model
        return MultKAN.loadckpt(path=f'{self.ckpt_path}/{model_id}')

    def checkout(self, model_id):
        '''
        check out an old version

        Args:
        -----
            model_id : str
                in format '{a}.{b}' where a is the round number, b is the version number in that round

        Returns:
        --------
            MultKAN

        '''
        return MultKAN.loadckpt(path=f'{self.ckpt_path}/{model_id}')

    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None

        '''
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])

    def update_grid(self, x):
        '''
        call update_grid_from_samples. This seems unnecessary but we retain it for the sake of classes that might inherit from MultKAN
        '''
        self.update_grid_from_samples(x)

    def initialize_grid_from_another_model(self, model, x):
        '''
        initialize grid from another model

        Args:
        -----
            model : MultKAN
                parent model
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[1,1], grid=5, k=3, seed=0)
        >>> print(model.act_fun[0].grid)
        >>> x = torch.linspace(-10,10,steps=101)[:,None]
        >>> model2 = KAN(width=[1,1], grid=10, k=3, seed=0)
        >>> model2.initialize_grid_from_another_model(model, x)
        >>> print(model2.act_fun[0].grid)
        '''
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

    def call(self, x, singularity_avoiding=False, y_th=10.):
        '''
        forward pass

        Args:
        -----
            x : 2D torch.tensor
                inputs
            singularity_avoiding : bool
                whether to avoid singularity for the symbolic branch
            y_th : float
                the threshold for singularity

        Returns:
        --------
            None

        '''
        x = tf.gather(x, self.input_id, axis=1)
        assert x.shape[1] == self.width_in[0]

        # cache data
        self.cache_data = x

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []
        # self.neurons_scale = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            # print(preacts, postacts_numerical, postspline)

            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding,
                                                                     y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic

            if self.save_act:
                self.subnode_actscale.append(tf.math.reduce_std(x, axis=0))

            # subnode affine transform
            x = self.subnode_scale[l][None, :] * x + self.subnode_bias[l][None, :]

            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic

                # self.neurons_scale.append(torch.mean(torch.abs(x), dim=0))
                # grid_reshape = self.act_fun[l].grid.reshape(self.width_out[l + 1], self.width_in[l], -1)
                input_range = tf.math.reduce_std(preacts, axis=0) + 0.1
                output_range_spline = tf.math.reduce_std(postacts_numerical,
                                                         axis=0)  # for training, only penalize the spline part
                output_range = tf.math.reduce_std(postacts,
                                                  axis=0)  # for visualization, include the contribution from both spline + symbolic
                # save edge_scale
                self.edge_actscale.append(output_range)

                self.acts_scale.append(output_range / input_range)
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts)
                self.spline_postacts.append(postacts)
                self.spline_postsplines.append(postspline)
                self.acts_premult.append(x)

            # multiplication
            dim_sum = self.width[l + 1][0]
            dim_mult = self.width[l + 1][1]

            if self.mult_homo == True:
                for i in range(self.mult_arity - 1):
                    if i == 0:
                        x_mult = x[:, dim_sum::self.mult_arity] * x[:, dim_sum + 1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:, dim_sum + i + 1::self.mult_arity]

            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + int(np.sum(self.mult_arity[l + 1][:j]))
                    for i in range(self.mult_arity[l + 1][j] - 1):
                        if i == 0:
                            x_mult_j = x[:, acml_id:acml_id + 1] * x[:, acml_id + 1:acml_id + 2]
                        else:
                            x_mult_j *= x[:, acml_id + i + 1:acml_id + i + 2]

                    x_mult = x_mult_j if j == 0 else tf.concat([x_mult, x_mult_j], axis=1)

            if dim_mult > 0:
                x = tf.concat([x[:, :dim_sum], x_mult], axis=1)

            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None, :] * x + self.node_bias[l][None, :]
            self.acts.append(tf.stop_gradient(x))

        return x

    def set_mode(self, l, i, j, mode, mask_n=None):
        if mode == "s":
            mask_n = 0.0
            mask_s = 1.0
        elif mode == "n":
            mask_n = 1.0
            mask_s = 0.0
        elif mode == "sn" or mode == "ns":
            if mask_n is None:
                mask_n = 1.0
            mask_s = 1.0
        else:
            mask_n = 0.0
            mask_s = 0.0

            self.act_fun[l].mask.data[i][j] = mask_n
            self.symbolic_fun[l].mask.data[j, i] = mask_s

    def fix_symbolic(self, l, i, j, fun_name, fit_params_bool=True, a_range=(-10, 10), b_range=(-10, 10), verbose=True,
                     random=False, log_history=True):
        '''
        set (l,i,j) activation to be symbolic (specified by fun_name)

       Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index
            fun_name : str
                function name
            fit_params_bool : bool
                obtaining affine parameters through fitting (True) or setting default values (False)
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of b
            verbose : bool
                If True, more information is printed.
            random : bool
                initialize affine parameteres randomly or as [1,0,1,0]
            log_history : bool
                indicate whether to log history when the function is called

        Returns:
        --------
            None or r2 (coefficient of determination)

        '''
        if not fit_params_bool:
            self.symbolic_fun[l].fix_symbolic(i, j, fun_name, verbose=verbose, random=random)
            r2 = None
        else:
            x = self.acts[l][:, i]
            mask = self.act_fun[l].mask
            y = self.spline_postacts[l][:, j, i]
            # y = self.postacts[l][:, j, i]
            r2 = self.symbolic_fun[l].fix_symbolic(i, j, fun_name, x, y, a_range=a_range, b_range=b_range, verbose=verbose)
            if mask[i, j] == 0:
                r2 = - 1e8
        self.set_mode(l, i, j, mode="s")

        if log_history:
            self.log_history('fix_symbolic')
        return r2

    def unfix_symbolic(self, l, i, j, log_history=True):
        '''
        unfix the (l,i,j) activation function.
        '''
        self.set_mode(l, i, j, mode="n")
        self.symbolic_fun[l].funs_name[j][i] = "0"
        if log_history:
            self.log_history('unfix_symbolic')

    def unfix_symbolic_all(self, log_history=True):
        '''
        unfix all activation functions.
        '''
        for l in range(len(self.width) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    self.unfix_symbolic(l, i, j, log_history)

    def get_range(self, l, i, j, verbose=True):
        '''
        Get the input range and output range of the (l,i,j) activation

        Args:
        -----
            l : int
                layer index
            i : int
                input neuron index
            j : int
                output neuron index

        Returns:
        --------
            x_min : float
                minimum of input
            x_max : float
                maximum of input
            y_min : float
                minimum of output
            y_max : float
                maximum of output

        '''
        x = self.spline_preacts[l][:, j, i]
        y = self.spline_postacts[l][:, j, i]
        x_min = tf.reduce_min(x).numpy()
        x_max = tf.reduce_max(x).numpy()
        y_min = tf.reduce_min(y).numpy()
        y_max = tf.reduce_max(y).numpy()
        if verbose:
            print(f"x range: [{x_min:.2f}, {x_max:.2f}]")
            print(f"y range: [{y_min:.2f}, {y_max:.2f}]")
        return x_min, x_max, y_min, y_max

    def plot(self, folder="./figures", beta=3, metric='backward', scale=0.5, tick=False, sample=False, in_vars=None,
             out_vars=None, title=None, varscale=1.0):
        '''
        plot KAN

        Args:
        -----
            folder : str
                the folder to store pngs
            beta : float
                positive number. control the transparency of each activation. transparency = tanh(beta*l1).
            mask : bool
                If True, plot with mask (need to run prune() first to obtain mask). If False (by default), plot all activation functions.
            mode : bool
                "supervised" or "unsupervised". If "supervised", l1 is measured by absolution value (not subtracting mean); if "unsupervised", l1 is measured by standard deviation (subtracting mean).
            scale : float
                control the size of the diagram
            in_vars: None or list of str
                the name(s) of input variables
            out_vars: None or list of str
                the name(s) of output variables
            title: None or str
                title
            varscale : float
                the size of input variables

        Returns:
        --------
            Figure

        '''
        global Symbol

        if not self.save_act:
            print('cannot plot since data are not saved. Set save_act=True first.')

        # forward to obtain activations
        if self.acts == None:
            if self.cache_data == None:
                raise Exception('model hasn\'t seen any data yet.')
            self.call(self.cache_data)

        if metric == 'backward':
            self.attribute()

        if not os.path.exists(folder):
            os.makedirs(folder)
        # matplotlib.use('Agg')
        depth = len(self.width) - 1
        for l in range(depth):
            w_large = 2.0
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    rank = tf.argsort(self.acts[l][:, i])
                    fig, ax = plt.subplots(figsize=(w_large, w_large))

                    num = rank.shape[0]

                    # print(self.width_in[l])
                    # print(self.width_out[l+1])
                    symbolic_mask = self.symbolic_fun[l].mask[j, i].numpy()
                    numeric_mask = self.act_fun[l].mask[i, j].numpy()
                    if symbolic_mask > 0. and numeric_mask > 0.:
                        color = 'purple'
                        alpha_mask = 1
                    if symbolic_mask > 0. and numeric_mask == 0.:
                        color = "red"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask > 0.:
                        color = "black"
                        alpha_mask = 1
                    if symbolic_mask == 0. and numeric_mask == 0.:
                        color = "white"
                        alpha_mask = 0

                    if tick == True:
                        ax.tick_params(axis="y", direction="in", pad=-22, labelsize=50)
                        ax.tick_params(axis="x", direction="in", pad=-15, labelsize=50)
                        x_min, x_max, y_min, y_max = self.get_range(l, i, j, verbose=False)
                        plt.xticks([x_min, x_max], ['%2.f' % x_min, '%2.f' % x_max])
                        plt.yticks([y_min, y_max], ['%2.f' % y_min, '%2.f' % y_max])
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    if alpha_mask == 1:
                        plt.gca().patch.set_edgecolor('black')
                    else:
                        plt.gca().patch.set_edgecolor('white')
                    plt.gca().patch.set_linewidth(1.5)
                    # plt.axis('off')

                    sorted_acts = tf.gather(self.acts[l][:, i], rank).numpy()
                    sorted_spline = tf.gather(self.spline_postacts[l][:, j, i], rank).numpy()
                    plt.plot(sorted_acts, sorted_spline, color=color, lw=5)
                    if sample == True:
                        plt.scatter(sorted_acts, sorted_spline, color=color, s=400 * scale ** 2)
                    plt.gca().spines[:].set_color(color)

                    plt.savefig(f'{folder}/sp_{l}_{i}_{j}.png', bbox_inches="tight", dpi=400)
                    plt.close()

        def score2alpha(score):
            return np.tanh(beta * score)

        if metric == 'forward_n':
            scores = self.acts_scale
        elif metric == 'forward_u':
            scores = self.edge_actscale
        elif metric == 'backward':
            scores = self.edge_scores
        else:
            raise Exception(f'metric = \'{metric}\' not recognized')

        alpha = [score2alpha(score.numpy()) for score in scores]

        # draw skeleton
        width = np.array(self.width)
        width_in = np.array(self.width_in)
        width_out = np.array(self.width_out)
        A = 1
        y0 = 0.3  # height: from input to pre-mult
        z0 = 0.1  # height: from pre-mult to post-mult (input of next layer)

        neuron_depth = len(width)
        min_spacing = A / np.maximum(np.max(width_out), 5)

        max_neuron = np.max(width_out)
        max_num_weights = np.max(width_in[:-1] * width_out[1:])
        y1 = 0.4 / np.maximum(max_num_weights, 5)  # size (height/width) of 1D function diagrams
        y2 = 0.15 / np.maximum(max_neuron, 5)  # size (height/width) of operations (sum and mult)

        fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * (neuron_depth - 1) * (y0 + z0)))
        # fig, ax = plt.subplots(figsize=(5,5*(neuron_depth-1)*y0))

        # -- Transformation functions
        DC_to_FC = ax.transData.transform
        FC_to_NFC = fig.transFigure.inverted().transform
        # -- Take data coordinates and transform them to normalized figure coordinates
        DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

        # plot scatters and lines
        for l in range(neuron_depth):

            n = width_in[l]

            # scatters
            for i in range(n):
                plt.scatter(1 / (2 * n) + i / n, l * (y0 + z0), s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

            # plot connections (input to pre-mult)
            if l < neuron_depth - 1:
                n_next = width_out[l + 1]
                N = n * n_next
                for i in range(n):
                    for j in range(n_next):
                        id_ = i * n_next + j

                        symbol_mask = self.symbolic_fun[l].mask[j, i].numpy()
                        numerical_mask = self.act_fun[l].mask[i, j].numpy()
                        if symbol_mask == 1. and numerical_mask > 0.:
                            color = 'purple'
                            alpha_mask = 1.
                        if symbol_mask == 1. and numerical_mask == 0.:
                            color = "red"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 1.:
                            color = "black"
                            alpha_mask = 1.
                        if symbol_mask == 0. and numerical_mask == 0.:
                            color = "white"
                            alpha_mask = 0.

                        plt.plot([1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                                 [l * (y0 + z0), l * (y0 + z0) + y0 / 2 - y1],
                                 color=color, lw=2 * scale, alpha=alpha[l][j][i] * alpha_mask)
                        plt.plot([1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                                 [l * (y0 + z0) + y0 / 2 + y1, l * (y0 + z0) + y0], color=color, lw=2 * scale,
                                 alpha=alpha[l][j][i] * alpha_mask)

            # plot connections (pre-mult to post-mult, post-mult = next-layer input)
            if l < neuron_depth - 1:
                n_in = width_out[l + 1]
                n_out = width_in[l + 1]
                mult_id = 0
                for i in range(n_in):
                    if i < width[l + 1][0]:
                        j = i
                    else:
                        if i == width[l + 1][0]:
                            if isinstance(self.mult_arity, int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l + 1][mult_id]
                            current_mult_arity = ma
                        if current_mult_arity == 0:
                            mult_id += 1
                            if isinstance(self.mult_arity, int):
                                ma = self.mult_arity
                            else:
                                ma = self.mult_arity[l + 1][mult_id]
                            current_mult_arity = ma
                        j = width[l + 1][0] + mult_id
                        current_mult_arity -= 1
                        # j = (i-width[l+1][0])//self.mult_arity + width[l+1][0]
                    plt.plot([1 / (2 * n_in) + i / n_in, 1 / (2 * n_out) + j / n_out],
                             [l * (y0 + z0) + y0, (l + 1) * (y0 + z0)], color='black', lw=2 * scale)

            plt.xlim(0, 1)
            plt.ylim(-0.1 * (y0 + z0), (neuron_depth - 1 + 0.1) * (y0 + z0))

        plt.axis('off')

        for l in range(neuron_depth - 1):
            # plot splines
            n = width_in[l]
            for i in range(n):
                n_next = width_out[l + 1]
                N = n * n_next
                for j in range(n_next):
                    id_ = i * n_next + j
                    im = plt.imread(f'{folder}/sp_{l}_{i}_{j}.png')
                    left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
                    right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
                    bottom = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 - y1])[1]
                    up = DC_to_NFC([0, l * (y0 + z0) + y0 / 2 + y1])[1]
                    newax = fig.add_axes([left, bottom, right - left, up - bottom])
                    # newax = fig.add_axes([1/(2*N)+id_/N-y1, (l+1/2)*y0-y1, y1, y1], anchor='NE')
                    newax.imshow(im, alpha=alpha[l][j][i])
                    newax.axis('off')

            # plot sum symbols
            N = n = width_out[l + 1]
            for j in range(n):
                id_ = j
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/sum_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, l * (y0 + z0) + y0 - y2])[1]
                up = DC_to_NFC([0, l * (y0 + z0) + y0 + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')

            # plot mult symbols
            N = n = width_in[l + 1]
            n_sum = width[l + 1][0]
            n_mult = width[l + 1][1]
            for j in range(n_mult):
                id_ = j + n_sum
                path = os.path.dirname(os.path.abspath(__file__)) + "/assets/img/mult_symbol.png"
                im = plt.imread(path)
                left = DC_to_NFC([1 / (2 * N) + id_ / N - y2, 0])[0]
                right = DC_to_NFC([1 / (2 * N) + id_ / N + y2, 0])[0]
                bottom = DC_to_NFC([0, (l + 1) * (y0 + z0) - y2])[1]
                up = DC_to_NFC([0, (l + 1) * (y0 + z0) + y2])[1]
                newax = fig.add_axes([left, bottom, right - left, up - bottom])
                newax.imshow(im)
                newax.axis('off')

        if in_vars != None:
            n = self.width_in[0]
            for i in range(n):
                if isinstance(in_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, f'${latex(in_vars[i])}$',
                                                 fontsize=40 * scale * varscale, horizontalalignment='center',
                                                 verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), -0.1, in_vars[i],
                                                 fontsize=40 * scale * varscale, horizontalalignment='center',
                                                 verticalalignment='center')

        if out_vars != None:
            n = self.width_in[-1]
            for i in range(n):
                if isinstance(out_vars[i], sympy.Expr):
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0 + z0) * (len(self.width) - 1) + 0.15,
                                                 f'${latex(out_vars[i])}$', fontsize=40 * scale * varscale,
                                                 horizontalalignment='center', verticalalignment='center')
                else:
                    plt.gcf().get_axes()[0].text(1 / (2 * (n)) + i / (n), (y0 + z0) * (len(self.width) - 1) + 0.15,
                                                 out_vars[i], fontsize=40 * scale * varscale,
                                                 horizontalalignment='center', verticalalignment='center')

        if title != None:
            plt.gcf().get_axes()[0].text(0.5, (y0 + z0) * (len(self.width) - 1) + 0.3, title, fontsize=40 * scale,
                                         horizontalalignment='center', verticalalignment='center')

    def reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff):
        '''
        Get regularization

        Args:
        -----
            reg_metric : the regularization metric
                'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient penalty strength
            lamb_coefdiff : float
                coefficient smoothness strength

        Returns:
        --------
            reg_ : torch.float

        Example
        -------
        >>> model = KAN(width=[2,3,1], grid=5, k=3, noise_scale=1.)
        >>> x = torch.rand(100,2)
        >>> model.get_act(x)
        >>> model.reg('edge_forward_spline_n', 1.0, 2.0, 1.0, 1.0)
        '''
        if reg_metric == 'edge_forward_spline_n':
            acts_scale = self.acts_scale_spline
            # print(self.acts_scale_spline)

        elif reg_metric == 'edge_forward_sum':
            acts_scale = self.acts_scale

        elif reg_metric == 'edge_forward_spline_u':
            acts_scale = self.edge_actscale

        elif reg_metric == 'edge_backward':
            acts_scale = self.edge_scores

        elif reg_metric == 'node_backward':
            acts_scale = self.node_attribute_scores

        else:
            raise Exception(f'reg_metric = {reg_metric} not recognized!')
        # print(len(acts_scale))
        reg_ = 0.
        for i in range(len(acts_scale)):
            vec = acts_scale[i]

            l1 = tf.reduce_sum(vec)
            p_row = vec / (tf.reduce_sum(vec, axis=1, keepdims=True) + 1)
            p_col = vec / (tf.reduce_sum(vec, axis=0, keepdims=True) + 1)
            entropy_row = - tf.reduce_mean(tf.reduce_sum(p_row * tf.math.log(p_row + 1e-4) / tf.math.log(2.0), axis=1))
            entropy_col = - tf.reduce_mean(tf.reduce_sum(p_col * tf.math.log(p_col + 1e-4) / tf.math.log(2.0), axis=0))
            # print(lamb_l1, l1, lamb_entropy, entropy_row, entropy_col)
            reg_ += lamb_l1 * l1 + lamb_entropy * (entropy_row + entropy_col)  # both l1 and entropy
            # print(reg_ )

        # regularize coefficient to encourage spline to be zero
        for i in range(len(self.act_fun)):
            coeff_l1 = tf.reduce_sum(tf.reduce_mean(tf.abs(self.act_fun[i].coef), axis=1))
            # print(self.act_fun[i].coef)
            # print(f"Shape of coef: {self.act_fun[i].coef.shape}")
            coeff_diff_l1 = tf.reduce_sum(
                tf.reduce_mean(tf.abs(tf.experimental.numpy.diff(self.act_fun[i].coef, axis=-1)), axis=1))
            # print(tf.experimental.numpy.diff(self.act_fun[i].coef, axis=-1).shape)
            # print(lamb_coef,coeff_l1,lamb_coefdiff,coeff_diff_l1)
            reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1
            # print(lamb_coef, coeff_l1, lamb_coefdiff, coeff_diff_l1)
            # print(i,reg_)

        # print(reg_)
        return reg_

    def get_reg(self, reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff):
        '''
        Get regularization. This seems unnecessary but in case a class wants to inherit this, it may want to rewrite get_reg, but not reg.
        '''
        return self.reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)

    def disable_symbolic_in_fit(self, lamb):
        '''
        during fitting, disable symbolic if either is true (lamb = 0, none of symbolic functions is active)
        '''
        old_save_act = self.save_act
        if lamb == 0.:
            self.save_act = False

        # skip symbolic if no symbolic is turned on
        depth = len(self.symbolic_fun)
        no_symbolic = True
        for l in range(depth):
            # Check if all elements in the mask are zero
            if tf.reduce_sum(tf.abs(self.symbolic_fun[l].mask)) > 0:
                no_symbolic = False
                break

        old_symbolic_enabled = self.symbolic_enabled

        if no_symbolic:
            self.symbolic_enabled = False

        return old_save_act, old_symbolic_enabled

    def get_params(self):
        '''
        Get parameters
        '''
        return [var for var in self.trainable_variables]

    def fit(self, dataset, opt="Adam", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
        metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None):
        """
        Training function for TensorFlow implementation of KAN model.

        Args:
            [Arguments are identical to the PyTorch version]

        Returns:
            results : dict
                Dictionary of training and testing losses, regularization values, and specified metrics.
        """
        self.learning_rate = lr
        if lamb > 0. and not self.save_act:
            print('Setting lamb=0. If you want to set lamb > 0, set self.save_act=True.')

        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn is None:
            loss_fn = lambda x, y: tf.reduce_mean(tf.square(x - y))
        elif loss_fn == "SparseCategoricalCrossentropy":
            loss_fn = lambda y_pred, y_true: -tf.reduce_mean(
                tf.reduce_sum(
                    tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(y_pred)[-1]) * tf.nn.log_softmax(
                        y_pred),
                    axis=-1
                )
            )
        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        # Optimizer selection
        if opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(
                              learning_rate=lr, history_size=10,
                              line_search_fn="strong_wolfe",
                              tolerance_grad=1e-32,
                              tolerance_change=1e-32,
                              tolerance_ys=1e-32)

        results = {
            'train_loss': [],
            'test_loss': [],
            'reg': []
        }
        if metrics:
            for metric in metrics:
                results[metric.__name__] = []

        train_inputs = dataset['train_input']
        train_labels = dataset['train_label']
        test_inputs = dataset['test_input']
        test_labels = dataset['test_label']

        batch_size = train_inputs.shape[0] if batch == -1 or batch > train_inputs.shape[0] else batch
        batch_size_test = test_inputs.shape[0] if batch == -1 or batch > test_inputs.shape[0] else batch

        # def closurer(train_sample, train_true):
        #     with tf.GradientTape() as tape:
        #         pred = self.call(train_sample, singularity_avoiding=singularity_avoiding, y_th=y_th)
        #         train_loss = loss_fn(pred, train_true)
        #         if self.save_act:
        #             if reg_metric == 'edge_backward':
        #                 self.attribute()
        #             if reg_metric == 'node_backward':
        #                 self.node_attribute()
        #             reg = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
        #         else:
        #             reg = tf.constant(0., dtype=tf.float32)
        #         objective = train_loss + lamb * reg
        #     grads = tape.gradient(objective, self.get_params())
        #     optimizer.apply_gradients(zip(grads, self.get_params()))
        #     return train_loss, reg
        def closure():
            pred = self.call(train_sample,
                             singularity_avoiding=singularity_avoiding,
                             y_th=y_th)
            local_train_loss = loss_fn(pred, train_true)
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                local_reg = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                local_reg = tf.constant(0., dtype=tf.float32)
            total_loss = local_train_loss + lamb * local_reg

            self.current_train_loss = local_train_loss
            self.current_reg = local_reg

            return total_loss

        if save_fig and not os.path.exists(img_folder):
            os.makedirs(img_folder)

        for step in pbar:
            if step == steps - 1 and old_save_act:
                self.save_act = True

            if save_fig and step % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True

            train_id = tf.random.shuffle(tf.range(tf.shape(train_inputs)[0]))[:batch_size]
            test_id = tf.random.shuffle(tf.range(tf.shape(test_inputs)[0]))[:batch_size_test]
            #train_indices = tf.convert_to_tensor(train_indices, dtype=tf.int32)

            train_sample = tf.gather(train_inputs, train_id)
            test_sample = tf.gather(test_inputs, test_id)
            train_true = tf.gather(train_labels, train_id)
            test_true = tf.gather(test_labels, test_id)

            # Update grid
            if step % grid_update_freq == 0 and step < stop_grid_update_step and update_grid and step >= start_grid_update_step:
                self.update_grid(train_sample)
            ###
            # Training step
            if opt == "LBFGS":
                optimizer.minimize(closure, var_list=self.trainable_variables)
                # 此时 self.current_train_loss 和 self.current_reg 已更新
                train_loss = self.current_train_loss
                reg = self.current_reg

            # if opt == "Adam":
            #     train_loss, reg = closurer(train_sample, train_true)

            ###
            # Evaluate on test set
            pred_test = self.call(test_sample, singularity_avoiding=singularity_avoiding, y_th=y_th)
            test_loss = loss_fn(pred_test, test_true)

            # Track metrics
            if metrics:
                for metric in metrics:
                    results[metric.__name__].append(metric().numpy())

            # Log results
            results['train_loss'].append(tf.sqrt(train_loss).numpy())
            results['test_loss'].append(tf.sqrt(test_loss).numpy())
            results['reg'].append(reg.numpy())

            if step % log == 0:
                if display_metrics is None:
                    pbar.set_description(
                        f"| train_loss: {results['train_loss'][-1]:.2e} | test_loss: {results['test_loss'][-1]:.2e} | reg: {results['reg'][-1]:.2e} |"
                    )
                else:
                    desc = ''
                    for metric_name in display_metrics:
                        desc += f"{metric_name}: {results[metric_name][-1]:.2e} | "
                    pbar.set_description(desc)

            # Save intermediate figures
            if save_fig and step % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title=f"Step {step}", beta=beta)
                plt.savefig(f"{img_folder}/{step}.jpg", bbox_inches='tight', dpi=200)
                plt.close()
                self.save_act = save_act

        self.log_history('fit')

        # Revert model state
        self.symbolic_enabled = old_symbolic_enabled
        return results

    def prune_node(self, threshold=2e-2, mode="auto", active_neurons_id=None, log_history=True):
        '''
        pruning nodes

        Args:
        -----
            threshold : float
                if the attribution score of a neuron is below the threshold, it is considered dead and will be removed
            mode : str
                'auto' or 'manual'. with 'auto', nodes are automatically pruned using threshold. with 'manual', active_neurons_id should be passed in.

        Returns:
        --------
            pruned network : MultKAN

        '''
        if self.acts == None:
            self.get_act()

        mask_up = [tf.ones(self.width_in[0], dtype=tf.float32)]
        mask_down = []
        active_neurons_up = [list(range(self.width_in[0]))]
        active_neurons_down = []
        num_sums = []
        num_mults = []
        mult_arities = [[]]

        if active_neurons_id != None:
            mode = "manual"
        # print(len(self.acts_scale))
        for i in range(len(self.acts_scale) - 1):

            mult_arity = []
            # print(self.node_scores[i + 1])
            if mode == "auto":
                self.attribute()
                overall_important_up = self.node_scores[i + 1] > threshold

            elif mode == "manual":
                overall_important_up = tf.zeros(self.width_in[i + 1], dtype=tf.bool)
                overall_important_up = tf.tensor_scatter_nd_update(
                    overall_important_up, tf.reshape(active_neurons_id[i], (-1, 1)),
                    tf.ones_like(active_neurons_id[i], dtype=tf.bool)
                )

            # print(self.mult_homo)
            num_sum = tf.reduce_sum(tf.cast(overall_important_up[:self.width[i + 1][0]], tf.int32)).numpy()
            num_mult = tf.reduce_sum(tf.cast(overall_important_up[self.width[i + 1][0]:], tf.int32)).numpy()
            if self.mult_homo == True:
                # overall_important_down = torch.cat([overall_important_up[:self.width[i+1][0]], (overall_important_up[self.width[i+1][0]:][None,:].expand(self.mult_arity,-1)).T.reshape(-1,)], dim=0)
                expanded = tf.repeat(overall_important_up[self.width[i + 1][0]:], self.mult_arity)
                overall_important_down = tf.concat([overall_important_up[:self.width[i + 1][0]], expanded], axis=0)

            else:
                overall_important_down = overall_important_up[:self.width[i + 1][0]]
                for j in range(overall_important_up[self.width[i + 1][0]:].shape[0]):
                    active_bool = overall_important_up[self.width[i + 1][0] + j]
                    arity = self.mult_arity[i + 1][j]
                    repeated_values = tf.fill([arity], active_bool)
                    overall_important_down = tf.concat([overall_important_down, repeated_values], axis=0)
                    if active_bool:
                        mult_arity.append(arity)

            num_sums.append(num_sum)
            num_mults.append(num_mult)

            mask_up.append(tf.cast(overall_important_up, tf.float32))
            mask_down.append(tf.cast(overall_important_down, tf.float32))

            active_neurons_up.append(tf.where(overall_important_up)[:, 0])
            active_neurons_down.append(tf.where(overall_important_down)[:, 0])

            mult_arities.append(mult_arity)

        active_neurons_down.append(list(range(self.width_out[-1])))
        mask_down.append(tf.ones(self.width_out[-1], dtype=tf.float32))

        if self.mult_homo == False:
            mult_arities.append(self.mult_arity[-1])

        self.mask_up = mask_up
        self.mask_down = mask_down

        # update act_fun[l].mask up
        for l in range(len(self.acts_scale) - 1):
            for i in range(self.width_in[l + 1]):
                if i not in active_neurons_up[l + 1].numpy():
                    self.remove_node(l + 1, i, mode='up', log_history=False)

            for i in range(self.width_out[l + 1]):
                if i not in active_neurons_down[l].numpy():
                    self.remove_node(l + 1, i, mode='down', log_history=False)

        current_weights = [w.numpy() for w in self.trainable_weights]
        model2 = MultKAN(width=copy.deepcopy(self.width), grid=self.grid, k=self.k,
                         base_fun=self.base_fun_name, mult_arity=self.mult_arity,
                         ckpt_path=self.ckpt_path, auto_save=True, first_init=False,
                         state_id=self.state_id, round=self.round)

        for old_w, new_w in zip(current_weights, model2.trainable_weights):
            new_w.assign(old_w)

        width_new = [self.width[0]]

        for i in range(len(self.acts_scale)):

            if i < len(self.acts_scale) - 1:
                num_sum = num_sums[i]
                num_mult = num_mults[i]

                # filtered_node_bias = tf.gather(model2.node_bias[i], active_neurons_up[i + 1])
                # filtered_node_scale = tf.gather(model2.node_scale[i], active_neurons_up[i + 1])
                # filtered_subnode_bias = tf.gather(model2.subnode_bias[i], active_neurons_down[i])
                # filtered_subnode_scale = tf.gather(model2.subnode_scale[i], active_neurons_down[i])

                model2.node_bias[i] = tf.gather(model2.node_bias[i], active_neurons_up[i + 1])
                model2.node_scale[i] = tf.gather(model2.node_scale[i], active_neurons_up[i + 1])
                model2.subnode_bias[i] = tf.gather(model2.subnode_bias[i], active_neurons_down[i])
                model2.subnode_scale[i] = tf.gather(model2.subnode_scale[i], active_neurons_down[i])

                model2.width[i + 1] = [num_sum, num_mult]

                model2.act_fun[i].out_dim_sum = num_sum
                model2.act_fun[i].out_dim_mult = num_mult

                model2.symbolic_fun[i].out_dim_sum = num_sum
                model2.symbolic_fun[i].out_dim_mult = num_mult

                width_new.append([num_sum, num_mult])

            model2.act_fun[i] = model2.act_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])
            model2.symbolic_fun[i] = self.symbolic_fun[i].get_subset(active_neurons_up[i], active_neurons_down[i])

        model2.cache_data = self.cache_data
        model2.acts = None

        width_new.append(self.width[-1])
        model2.width = width_new

        if self.mult_homo == False:
            model2.mult_arity = mult_arities

        if log_history:
            self.log_history('prune_node')
            model2.state_id += 1

        return model2

    def prune_edge(self, threshold=3e-2, log_history=True):
        '''
        pruning edges

        Args:
        -----
            threshold : float
                if the attribution score of an edge is below the threshold, it is considered dead and will be set to zero.

        Returns:
        --------
            pruned network : MultKAN

        '''
        if self.acts == None:
            self.get_act()

        for i in range(len(self.width) - 1):
            # self.act_fun[i].mask.data = ((self.acts_scale[i] > threshold).permute(1,0)).float()
            old_mask = self.act_fun[i].mask
            new_mask = tf.cast(tf.transpose(self.edge_scores[i] > threshold), dtype=tf.float32) * old_mask
            self.act_fun[i].mask.assign(new_mask)

        if log_history:
            self.log_history('fix_symbolic')

    def prune(self, node_th=1e-2, edge_th=3e-2):
        '''
        prune (both nodes and edges)

        Args:
        -----
            node_th : float
                if the attribution score of a node is below node_th, it is considered dead and will be set to zero.
            edge_th : float
                if the attribution score of an edge is below node_th, it is considered dead and will be set to zero.

        Returns:
        --------
            pruned network : MultKAN

        '''
        if self.acts == None:
            self.get_act()

        self = self.prune_node(node_th, log_history=False)
        # self.prune_node(node_th, log_history=False)
        self.call(self.cache_data)
        self.attribute()
        self.prune_edge(edge_th, log_history=False)
        self.log_history('prune')
        return self

    def prune_input(self, threshold=1e-2, active_inputs=None, log_history=True):
        '''
        prune inputs

        Args:
        -----
            threshold : float
                if the attribution score of the input feature is below threshold, it is considered irrelevant.
            active_inputs : None or list
                if a list is passed, the manual mode will disregard attribution score and prune as instructed.

        Returns:
        --------
            pruned network : MultKAN

        '''
        if active_inputs == None:
            self.attribute()
            input_score = self.node_scores[0]
            input_mask = input_score > threshold
            print('keep:', input_mask.numpy().tolist())
            input_id = tf.where(input_mask)[:, 0]

        else:
            input_id = tf.constant(active_inputs, dtype=tf.int32)

        current_weights = [w.numpy() for w in self.trainable_weights]
        model2 = MultKAN(width=self.width.copy(), grid=self.grid, k=self.k,
                         base_fun=self.base_fun, mult_arity=self.mult_arity, ckpt_path=self.ckpt_path,
                         auto_save=True, first_init=False, state_id=self.state_id, round=self.round)
        for old_w, new_w in zip(current_weights, model2.trainable_weights):
            new_w.assign(old_w)

        model2.act_fun[0] = model2.act_fun[0].get_subset(input_id, tf.range(self.width_out[1]))
        model2.symbolic_fun[0] = self.symbolic_fun[0].get_subset(input_id, tf.range(self.width_out[1]))

        model2.cache_data = self.cache_data
        model2.acts = None

        model2.width[0] = [len(input_id.numpy()), 0]
        model2.input_id = input_id

        if log_history:
            self.log_history('prune_input')
            model2.state_id += 1

        return model2

    def remove_edge(self, l, i, j, log_history=True):
        '''
        remove activtion phi(l,i,j) (set its mask to zero)
        '''
        self.act_fun[l].mask = tf.tensor_scatter_nd_update(
            self.act_fun[l].mask,
            indices=[[i, j]],
            updates=[0.0]
        )
        if log_history:
            self.log_history('remove_edge')

    def remove_node(self, l, i, mode='all', log_history=True):
        '''
        remove neuron (l,i) (set the masks of all incoming and outgoing activation functions to zero)
        '''
        if mode == 'down':
            self.act_fun[l - 1].mask = tf.tensor_scatter_nd_update(
                self.act_fun[l - 1].mask,
                indices=[[row, i] for row in range(self.act_fun[l - 1].mask.shape[0])],
                updates=[0.0] * self.act_fun[l - 1].mask.shape[0]
            )
            self.symbolic_fun[l - 1].mask = tf.tensor_scatter_nd_update(
                self.symbolic_fun[l - 1].mask,
                indices=[[i, col] for col in range(self.symbolic_fun[l - 1].mask.shape[1])],
                updates=[0.0] * self.symbolic_fun[l - 1].mask.shape[1]
            )

        elif mode == 'up':
            self.act_fun[l].mask = tf.tensor_scatter_nd_update(
                self.act_fun[l].mask,
                indices=[[i, col] for col in range(self.act_fun[l].mask.shape[1])],
                updates=[0.0] * self.act_fun[l].mask.shape[1]
            )
            self.symbolic_fun[l].mask = tf.tensor_scatter_nd_update(
                self.symbolic_fun[l].mask,
                indices=[[row, i] for row in range(self.symbolic_fun[l].mask.shape[0])],
                updates=[0.0] * self.symbolic_fun[l].mask.shape[0]
            )

        else:
            self.remove_node(l, i, mode='up', log_history=False)
            self.remove_node(l, i, mode='down', log_history=False)

        if log_history:
            self.log_history('remove_node')

    def attribute(self, l=None, i=None, out_score=None, plot=True):
        '''
        get attribution scores

        Args:
        -----
            l : None or int
                layer index
            i : None or int
                neuron index
            out_score : None or 1D torch.float
                specify output scores
            plot : bool
                when plot = True, display the bar show

        Returns:
        --------
            attribution scores

        '''
        # output (out_dim, in_dim)

        if l != None:
            self.attribute()
            out_score = self.node_scores[l]

        if self.acts == None:
            self.get_act()

        def score_node2subnode(node_score, width, mult_arity, out_dim):

            assert np.sum(width) == node_score.shape[1]
            if isinstance(mult_arity, int):
                n_subnode = width[0] + mult_arity * width[1]
            else:
                n_subnode = width[0] + int(np.sum(mult_arity))

            # subnode_score_leaf = torch.zeros(out_dim, n_subnode).requires_grad_(True)
            # subnode_score = subnode_score_leaf.clone()
            # subnode_score[:,:width[0]] = node_score[:,:width[0]]
            subnode_score = node_score[:, :width[0]]
            if isinstance(mult_arity, int):
                # subnode_score[:,width[0]:] = node_score[:,width[0]:][:,:,None].expand(out_dim, node_score[width[0]:].shape[0], mult_arity).reshape(out_dim,-1)
                expanded_score = tf.reshape(
                    tf.repeat(node_score[:, width[0]:][:, :, None], mult_arity, axis=2),
                    [out_dim, -1]
                )
                subnode_score = tf.concat([subnode_score, expanded_score], axis=1)
            else:
                acml = width[0]
                for idx, m_arity in enumerate(mult_arity):
                    expanded_score = tf.tile(tf.expand_dims(node_score[:, width[0] + idx], axis=1), [1, m_arity])
                    subnode_score = tf.concat([subnode_score, expanded_score], axis=1)
                    acml += m_arity
            return subnode_score

        node_scores = []
        subnode_scores = []
        edge_scores = []

        l_query = l
        if l == None:
            l_end = self.depth
        else:
            l_end = l

        # back propagate from the queried layer
        out_dim = self.width_in[l_end]
        if out_score == None:
            node_score = tf.eye(out_dim, dtype=tf.float32)
        else:
            node_score = tf.linalg.diag(out_score)
        node_scores.append(node_score)

        device = self.act_fun[0].grid.device

        for l in range(l_end, 0, -1):

            # node to subnode
            if isinstance(self.mult_arity, int):
                subnode_score = score_node2subnode(node_score, self.width[l], self.mult_arity, out_dim=out_dim)
            else:
                mult_arity = self.mult_arity[l]
                # subnode_score = score_node2subnode(node_score, self.width[l], mult_arity)
                subnode_score = score_node2subnode(node_score, self.width[l], mult_arity, out_dim=out_dim)

            subnode_scores.append(subnode_score)
            # subnode to edge
            # print(self.edge_actscale[l-1].device, subnode_score.device, self.subnode_actscale[l-1].device)
            # edge_score = torch.einsum('ij,ki,i->kij', self.edge_actscale[l-1], subnode_score.to(device), 1/(self.subnode_actscale[l-1]+1e-4))
            edge_score = tf.einsum(
                'ij,ki,i->kij',
                self.edge_actscale[l - 1],
                subnode_score,
                1.0 / (self.subnode_actscale[l - 1] + 1e-4)
            )
            edge_scores.append(edge_score)

            # edge to node
            node_score = tf.reduce_sum(edge_score, axis=1)
            node_scores.append(node_score)

        self.node_scores_all = list(reversed(node_scores))
        self.edge_scores_all = list(reversed(edge_scores))
        self.subnode_scores_all = list(reversed(subnode_scores))

        self.node_scores = [tf.reduce_mean(l, axis=0) for l in self.node_scores_all]
        self.edge_scores = [tf.reduce_mean(l, axis=0) for l in self.edge_scores_all]
        self.subnode_scores = [tf.reduce_mean(l, axis=0) for l in self.subnode_scores_all]

        # return
        if l_query != None:
            if i == None:
                return self.node_scores_all[0]
            else:

                # plot
                if plot:
                    in_dim = self.width_in[0]
                    plt.figure(figsize=(1 * in_dim, 3))
                    plt.bar(range(in_dim), self.node_scores_all[0][i].numpy())
                    plt.xticks(range(in_dim));

                return self.node_scores_all[0][i]

    def node_attribute(self):
        self.node_attribute_scores = []
        for l in range(1, self.depth + 1):
            node_attr = self.attribute(l)
            self.node_attribute_scores.append(node_attr)

    def suggest_symbolic(self, l, i, j, a_range=(-10, 10), b_range=(-10, 10), lib=None, topk=5, verbose=True,
                         r2_loss_fun=lambda x: np.log2(1 + 1e-5 - x), c_loss_fun=lambda x: x, weight_simple=0.8):
        '''
        suggest symbolic function

        Args:
        -----
            l : int
                layer index
            i : int
                neuron index in layer l
            j : int
                neuron index in layer j
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            topk : int
                the number of top functions displayed
            verbose : bool
                if verbose = True, print more information
            r2_loss_fun : functoon
                function : r2 -> "bits"
            c_loss_fun : fun
                function : c -> 'bits'
            weight_simple : float
                the simplifty weight: the higher, more prefer simplicity over performance


        Returns:
        --------
            best_name (str), best_fun (function), best_r2 (float), best_c (float)

        '''
        r2s = []
        cs = []

        if lib == None:
            symbolic_lib = SYMBOLIC_LIB
        else:
            symbolic_lib = {}
            for item in lib:
                symbolic_lib[item] = SYMBOLIC_LIB[item]

        # getting r2 and complexities
        for (name, content) in symbolic_lib.items():
            r2 = self.fix_symbolic(l, i, j, name, a_range=a_range, b_range=b_range, verbose=False, log_history=False)
            if r2 == -1e8:  # zero function
                r2s.append(-1e8)
            else:
                r2s.append(r2.item())
                self.unfix_symbolic(l, i, j, log_history=False)
            c = content[2]
            cs.append(c)

        r2s = np.array(r2s)
        cs = np.array(cs)
        r2_loss = r2_loss_fun(r2s).astype('float')
        cs_loss = c_loss_fun(cs)

        loss = weight_simple * cs_loss + (1 - weight_simple) * r2_loss

        sorted_ids = np.argsort(loss)[:topk]
        r2s = r2s[sorted_ids][:topk]
        cs = cs[sorted_ids][:topk]
        r2_loss = r2_loss[sorted_ids][:topk]
        cs_loss = cs_loss[sorted_ids][:topk]
        loss = loss[sorted_ids][:topk]

        topk = np.minimum(topk, len(symbolic_lib))

        if verbose == True:
            # print results in a dataframe
            results = {}
            results['function'] = [list(symbolic_lib.items())[sorted_ids[i]][0] for i in range(topk)]
            results['fitting r2'] = r2s[:topk]
            results['r2 loss'] = r2_loss[:topk]
            results['complexity'] = cs[:topk]
            results['complexity loss'] = cs_loss[:topk]
            results['total loss'] = loss[:topk]

            df = pd.DataFrame(results)
            print(df)

        best_name = list(symbolic_lib.items())[sorted_ids[0]][0]
        best_fun = list(symbolic_lib.items())[sorted_ids[0]][1]
        best_r2 = r2s[0]
        best_c = cs[0]

        return best_name, best_fun, best_r2, best_c;

    def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1, weight_simple=0.8,
                      r2_threshold=0.0):
        '''
        automatic symbolic regression for all edges

        Args:
        -----
            a_range : tuple
                search range of a
            b_range : tuple
                search range of b
            lib : list of str
                library of candidate symbolic functions
            verbose : int
                larger verbosity => more verbosity
            weight_simple : float
                a weight that prioritizies simplicity (low complexity) over performance (high r2) - set to 0.0 to ignore complexity
            r2_threshold : float
                If r2 is below this threshold, the edge will not be fixed with any symbolic function - set to 0.0 to ignore this threshold
        Returns:
        --------
            None
        '''
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    if self.symbolic_fun[l].mask[j, i] > 0. and self.act_fun[l].mask[i][j] == 0.:
                        print(f'skipping ({l},{i},{j}) since already symbolic')
                    elif self.symbolic_fun[l].mask[j, i] == 0. and self.act_fun[l].mask[i][j] == 0.:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                        print(f'fixing ({l},{i},{j}) with 0')
                    else:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=lib,
                                                                 verbose=False, weight_simple=weight_simple)
                        if r2 >= r2_threshold:
                            self.fix_symbolic(l, i, j, name, verbose=verbose > 1, log_history=False)
                            if verbose >= 1:
                                print(f'fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}')
                        else:
                            print(
                                f'For ({l},{i},{j}) the best fit was {name}, but r^2 = {r2} and this is lower than {r2_threshold}. This edge was omitted, keep training or try a different threshold.')

        self.log_history('auto_symbolic')

    def symbolic_formula(self, var=None, normalizer=None, output_normalizer=None):
        '''
        get symbolic formula

        Args:
        -----
            var : None or a list of sympy expression
                input variables
            normalizer : [mean, std]
            output_normalizer : [mean, std]

        Returns:
        --------
            None

        '''

        symbolic_acts = []
        symbolic_acts_premult = []
        x = []

        def ex_round(ex1, n_digit):
            ex2 = ex1
            for a in sympy.preorder_traversal(ex1):
                if isinstance(a, sympy.Float):
                    ex2 = ex2.subs(a, round(a, n_digit))
            return ex2

        # define variables
        if var == None:
            for ii in range(1, self.width[0][0] + 1):
                exec(f"x{ii} = sympy.Symbol('x_{ii}')")
                exec(f"x.append(x{ii})")
        elif isinstance(var[0], sympy.Expr):
            x = var
        else:
            x = [sympy.symbols(var_) for var_ in var]

        x0 = x

        if normalizer != None:
            mean = normalizer[0]
            std = normalizer[1]
            x = [(x[i] - mean[i]) / std[i] for i in range(len(x))]

        symbolic_acts.append(x)

        for l in range(len(self.width_in) - 1):
            num_sum = self.width[l + 1][0]
            num_mult = self.width[l + 1][1]
            y = []
            for j in range(self.width_out[l + 1]):
                yj = 0.
                for i in range(self.width_in[l]):
                    a, b, c, d = self.symbolic_fun[l].affine[j, i]
                    sympy_fun = self.symbolic_fun[l].funs_sympy[j][i]
                    try:
                        yj += c * sympy_fun(a * x[i] + b) + d
                    except:
                        print('make sure all activations need to be converted to symbolic formulas first!')
                        return
                yj = self.subnode_scale[l][j] * yj + self.subnode_bias[l][j]
                if simplify == True:
                    y.append(sympy.simplify(yj))
                else:
                    y.append(yj)

            symbolic_acts_premult.append(y)

            mult = []
            for k in range(num_mult):
                if isinstance(self.mult_arity, int):
                    mult_arity = self.mult_arity
                else:
                    mult_arity = self.mult_arity[l + 1][k]
                for i in range(mult_arity - 1):
                    if i == 0:
                        mult_k = y[num_sum + 2 * k] * y[num_sum + 2 * k + 1]
                    else:
                        mult_k = mult_k * y[num_sum + 2 * k + i + 1]
                mult.append(mult_k)

            y = y[:num_sum] + mult

            for j in range(self.width_in[l + 1]):
                y[j] = self.node_scale[l][j] * y[j] + self.node_bias[l][j]

            x = y
            symbolic_acts.append(x)

        if output_normalizer != None:
            output_layer = symbolic_acts[-1]
            means = output_normalizer[0]
            stds = output_normalizer[1]

            assert len(output_layer) == len(means), 'output_normalizer does not match the output layer'
            assert len(output_layer) == len(stds), 'output_normalizer does not match the output layer'

            output_layer = [(output_layer[i] * stds[i] + means[i]) for i in range(len(output_layer))]
            symbolic_acts[-1] = output_layer

        self.symbolic_acts = [[symbolic_acts[l][i] for i in range(len(symbolic_acts[l]))] for l in
                              range(len(symbolic_acts))]
        self.symbolic_acts_premult = [[symbolic_acts_premult[l][i] for i in range(len(symbolic_acts_premult[l]))] for l
                                      in range(len(symbolic_acts_premult))]

        out_dim = len(symbolic_acts[-1])
        # return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

        if simplify:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0
        else:
            return [symbolic_acts[-1][i] for i in range(len(symbolic_acts[-1]))], x0

    def module(self, start_layer, chain):
        '''
        specify network modules

        Args:
        -----
            start_layer : int
                the earliest layer of the module
            chain : str
                specify neurons in the module

        Returns:
        --------
            None
        '''
        # chain = '[-1]->[-1,-2]->[-1]->[-1]'
        groups = chain.split('->')
        n_total_layers = len(groups) // 2
        # start_layer = 0

        for l in range(n_total_layers):
            current_layer = cl = start_layer + l
            id_in = [int(i) for i in groups[2 * l][1:-1].split(',')]
            id_out = [int(i) for i in groups[2 * l + 1][1:-1].split(',')]

            in_dim = self.width_in[cl]
            out_dim = self.width_out[cl + 1]
            id_in_other = list(set(range(in_dim)) - set(id_in))
            id_out_other = list(set(range(out_dim)) - set(id_out))
            self.act_fun[cl].mask.data[np.ix_(id_in_other, id_out)] = 0.
            self.act_fun[cl].mask.data[np.ix_(id_in, id_out_other)] = 0.
            self.symbolic_fun[cl].mask.data[np.ix_(id_out, id_in_other)] = 0.
            self.symbolic_fun[cl].mask.data[np.ix_(id_out_other, id_in)] = 0.

        self.log_history('module')

    def get_act(self, x=None):
        '''
        collect intermidate activations
        '''
        if isinstance(x, dict):
            x = x['train_input']
        if x == None:
            if self.cache_data != None:
                x = self.cache_data
            else:
                raise Exception("missing input data x")
        save_act = self.save_act
        self.save_act = True
        self.call(x)
        self.save_act = save_act

    def get_fun(self, l, i, j):
        '''
        get function (l,i,j)
        '''
        inputs = self.spline_preacts[l][:, j, i].cpu().detach().numpy()
        outputs = self.spline_postacts[l][:, j, i].cpu().detach().numpy()
        # they are not ordered yet
        rank = np.argsort(inputs)
        inputs = inputs[rank]
        outputs = outputs[rank]
        plt.figure(figsize=(3, 3))
        plt.plot(inputs, outputs, marker="o")
        return inputs, outputs

    def history(self, k='all'):
        '''
        get history
        '''
        with open(self.ckpt_path + '/history.txt', 'r') as f:
            data = f.readlines()
            n_line = len(data)
            if k == 'all':
                k = n_line

            data = data[-k:]
            for line in data:
                print(line[:-1])


KAN = MultKAN