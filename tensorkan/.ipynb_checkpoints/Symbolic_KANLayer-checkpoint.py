import tensorflow as tf
import numpy as np
import sympy
from .utils import *



class Symbolic_KANLayer(tf.keras.layers.Layer):
    '''
    KANLayer class

    Attributes:
    -----------
        in_dim : int
            input dimension
        out_dim : int
            output dimension
        funs : 2D array of torch functions (or lambda functions)
            symbolic functions (torch)
        funs_avoid_singularity : 2D array of torch functions (or lambda functions) with singularity avoiding
        funs_name : 2D arry of str
            names of symbolic functions
        funs_sympy : 2D array of sympy functions (or lambda functions)
            symbolic functions (sympy)
        affine : 3D array of floats
            affine transformations of inputs and outputs
    '''
    def __init__(self, in_dim=3, out_dim=2):
        '''
        initialize a Symbolic_KANLayer (activation functions are initialized to be identity functions)
        
        Args:
        -----
            in_dim : int
                input dimension
            out_dim : int
                output dimension
            device : str
                device
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=3)
        >>> len(sb.funs), len(sb.funs[0])
        '''
        super(Symbolic_KANLayer, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.mask = tf.Variable(tf.zeros((out_dim, in_dim)), trainable=False)
        # torch
        self.funs = [[lambda x: x*0. for i in range(self.in_dim)] for j in range(self.out_dim)]
        self.funs_avoid_singularity = [[lambda x, y_th: ((), x*0.) for i in range(self.in_dim)] for j in range(self.out_dim)]
        # name
        self.funs_name = [['0' for i in range(self.in_dim)] for j in range(self.out_dim)]
        # sympy
        self.funs_sympy = [[lambda x: tf.zeros_like(x) for _ in range(self.in_dim)] for _ in range(self.out_dim)]
        ### make funs_name the only parameter, and make others as the properties of funs_name?
        
        self.affine = tf.Variable(tf.zeros((out_dim, in_dim, 4)), trainable=True)
        # c*f(a*x+b)+d

    def call(self, x, singularity_avoiding=False, y_th=10.0):
        '''
        forward
        
        Args:
        -----
            x : 2D array
                inputs, shape (batch, input dimension)
            singularity_avoiding : bool
                if True, funs_avoid_singularity is used; if False, funs is used. 
            y_th : float
                the singularity threshold
            
        Returns:
        --------
            y : 2D array
                outputs, shape (batch, output dimension)
            postacts : 3D array
                activations after activation functions but before being summed on nodes
        
        Example
        -------
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=5)
        >>> x = torch.normal(0,1,size=(100,3))
        >>> y, postacts = sb(x)
        >>> y.shape, postacts.shape
        (torch.Size([100, 5]), torch.Size([100, 5, 3]))
        '''
        
        batch = tf.shape(x)[0]
        postacts = []

        for i in range(self.in_dim):
            postacts_ = []
            for j in range(self.out_dim):
                if singularity_avoiding:
                    xij = self.affine[j, i, 2] * self.funs_avoid_singularity[j][i](self.affine[j, i, 0] * x[:, i:i+1] + self.affine[j, i, 1], y_th)[1] + self.affine[j, i, 3]
                else:
                    xij = self.affine[j, i, 2] * self.funs[j][i](self.affine[j, i, 0] * x[:, i:i+1] + self.affine[j, i, 1]) + self.affine[j, i, 3]
                postacts_.append(self.mask[j, i] * xij)
            postacts.append(tf.stack(postacts_))

        postacts = tf.stack(postacts, axis=0)
        postacts = tf.transpose(postacts, [2, 1, 0, 3])[:, :, :, 0]
        y = tf.reduce_sum(postacts, axis=2)

        return y, postacts
        
    def get_subset(self, in_id, out_id):
        '''
        get a smaller Symbolic_KANLayer from a larger Symbolic_KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : Symbolic_KANLayer
         
        Example
        -------
        >>> sb_large = Symbolic_KANLayer(in_dim=10, out_dim=10)
        >>> sb_small = sb_large.get_subset([0,9],[1,2,3])
        >>> sb_small.in_dim, sb_small.out_dim
        '''
        sbb = Symbolic_KANLayer(len(in_id), len(out_id))
        sbb.in_dim = len(in_id)
        sbb.out_dim = len(out_id)
        sbb.mask.assign(self.mask.numpy()[np.ix_(out_id, in_id)])
        sbb.funs = [[self.funs[j][i] for i in in_id] for j in out_id]
        sbb.funs_avoid_singularity = [[self.funs_avoid_singularity[j][i] for i in in_id] for j in out_id]
        sbb.funs_sympy = [[self.funs_sympy[j][i] for i in in_id] for j in out_id]
        sbb.funs_name = [[self.funs_name[j][i] for i in in_id] for j in out_id]
        sbb.affine.assign(self.affine.numpy()[np.ix_(out_id, in_id, [0, 1, 2, 3])])
        return sbb
    
    def fix_symbolic(self, i, j, fun_name, x=None, y=None, random=False, a_range=(-10, 10), b_range=(-10, 10), verbose=True):
        '''
        fix an activation function to be symbolic
        
        Args:
        -----
            i : int
                the id of input neuron
            j : int 
                the id of output neuron
            fun_name : str
                the name of the symbolic functions
            x : 1D array
                preactivations
            y : 1D array
                postactivations
            a_range : tuple
                sweeping range of a
            b_range : tuple
                sweeping range of a
            verbose : bool
                print more information if True
            
        Returns:
        --------
            r2 (coefficient of determination)
            
        Example 1
        ---------
        >>> # when x & y are not provided. Affine parameters are set to a = 1, b = 0, c = 1, d = 0
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> sb.fix_symbolic(2,1,'sin')
        >>> print(sb.funs_name)
        >>> print(sb.affine)
        
        Example 2
        ---------
        >>> # when x & y are provided, fit_params() is called to find the best fit coefficients
        >>> sb = Symbolic_KANLayer(in_dim=3, out_dim=2)
        >>> batch = 100
        >>> x = torch.linspace(-1,1,steps=batch)
        >>> noises = torch.normal(0,1,(batch,)) * 0.02
        >>> y = 5.0*torch.sin(3.0*x + 2.0) + 0.7 + noises
        >>> sb.fix_symbolic(2,1,'sin',x,y)
        >>> print(sb.funs_name)
        >>> print(sb.affine[1,2,:].data)
        '''
        if isinstance(fun_name, str):
            fun = SYMBOLIC_LIB[fun_name][0]
            fun_sympy = SYMBOLIC_LIB[fun_name][1]
            fun_avoid_singularity = SYMBOLIC_LIB[fun_name][3]
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = fun_name

            if x is None or y is None:
            # Initialize from just fun
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                if not random:
                    self.affine[j, i].assign(tf.constant([1.0, 0.0, 1.0, 0.0], dtype=tf.float32))
                else:
                    self.affine[j, i].assign(tf.random.uniform([4], minval=-1.0, maxval=1.0, dtype=tf.float32))
                return None
            else:
                # Initialize from x & y and fun
                params, r2 = fit_params(x, y, fun, a_range=a_range, b_range=b_range, verbose=verbose)
                self.funs[j][i] = fun
                self.funs_avoid_singularity[j][i] = fun_avoid_singularity
                self.affine[j, i].assign(params)
                return r2
        else:
                # If fun_name itself is a function
            fun = fun_name
            fun_sympy = fun_name
            self.funs_sympy[j][i] = fun_sympy
            self.funs_name[j][i] = "anonymous"

            self.funs[j][i] = fun
            self.funs_avoid_singularity[j][i] = fun
            if not random:
                self.affine[j, i].assign(tf.constant([1.0, 0.0, 1.0, 0.0], dtype=tf.float32))
            else:
                self.affine[j, i].assign(tf.random.uniform([4], minval=-1.0, maxval=1.0, dtype=tf.float32))
            return None
        
    def swap(self, i1, i2, mode='in'):
        """
        Swap the i1 neuron with the i2 neuron in input (if mode == 'in') or output (if mode == 'out').
        """
        def swap_list_(data, i1, i2, mode='in'):
            if mode == 'in':
                for j in range(self.out_dim):
                    data[j][i1], data[j][i2] = data[j][i2], data[j][i1]
            elif mode == 'out':
                data[i1], data[i2] = data[i2], data[i1]

        def swap_tensor(tensor, i1, i2, mode='in'):
            if mode == 'in':
                temp = tf.identity(tensor[:, i1])
                tensor = tensor.numpy()  # Convert to NumPy for in-place modification
                tensor[:, i1] = tensor[:, i2]
                tensor[:, i2] = temp
                return tf.convert_to_tensor(tensor)
            elif mode == 'out':
                temp = tf.identity(tensor[i1])
                tensor = tensor.numpy()
                tensor[i1] = tensor[i2]
                tensor[i2] = temp
                return tf.convert_to_tensor(tensor)

        swap_list_(self.funs_name, i1, i2, mode)
        swap_list_(self.funs_sympy, i1, i2, mode)
        swap_list_(self.funs_avoid_singularity, i1, i2, mode)
        self.affine = swap_tensor(self.affine, i1, i2, mode)
        self.mask = swap_tensor(self.mask, i1, i2, mode)

