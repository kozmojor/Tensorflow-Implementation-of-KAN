import tensorflow as tf
import numpy as np



def f_inv(x, y_th):
    x_th = 1 / y_th
    condition = tf.abs(x) < x_th
    safe_inverse = tf.where(condition, y_th / x_th * x, tf.math.divide_no_nan(1, x))
    return safe_inverse
def f_inv2(x, y_th):
    x_th = 1 / tf.pow(y_th, 1/2)
    condition = tf.abs(x) < x_th
    safe_inverse = tf.where(condition, y_th, tf.math.divide_no_nan(1, tf.pow(x, 2)))
    return safe_inverse
def f_sqrt(x, y_th):
    x_th = 1 / tf.square(y_th)
    condition = tf.abs(x) < x_th
    safe_sqrt = tf.where(
        condition,
        x_th / y_th * x,
        tf.math.sqrt(tf.abs(x)) * tf.sign(x)
    )
    return safe_sqrt
def f_log(x, y_th):
    x_th = tf.exp(-y_th)
    condition = tf.abs(x) < x_th
    safe_log = tf.where(
        condition,
        -y_th,
        tf.math.log(tf.abs(x))
    )
    return safe_log
def f_tan(x, y_th):
    clip = x % tf.constant(tf.math.pi)
    delta = tf.constant(tf.math.pi / 2) - tf.math.atan(y_th)
    safe_tan = tf.where(
        tf.abs(clip - tf.constant(tf.math.pi / 2)) < delta,
        -y_th / delta * (clip - tf.constant(tf.math.pi / 2)),
        tf.math.tan(clip)
    )
    return safe_tan
def f_exp(x, y_th):
    x_th = tf.math.log(y_th)
    safe_exp = tf.where(x > x_th, y_th, tf.exp(x))
    return safe_exp

def create_dataset(f,
                   n_var=2,
                   f_mode='col',
                   ranges=[-1, 1],
                   train_num=1000,
                   test_num=1000,
                   normalize_input=False,
                   normalize_label=False,
                   seed=0):
    """
    Generate synthetic datasets with symbolic functions in TensorFlow.

    Args:
        f: Function
            Symbolic formula for generating data.
        n_var: int
            Number of input variables.
        f_mode: str
            'col' for column-wise or 'row' for row-wise function.
        ranges: list or array
            Range of inputs.
        train_num: int
            Number of training samples.
        test_num: int
            Number of test samples.
        normalize_input: bool
            Normalize input if True.
        normalize_label: bool
            Normalize labels if True.
        seed: int
            Random seed.

    Returns:
        dataset: dict
            A dictionary containing train/test input and labels.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    ranges = np.array(ranges * n_var).reshape(n_var, 2) if len(np.array(ranges).shape) == 1 else np.array(ranges)

    train_input = tf.convert_to_tensor(
        np.random.uniform(ranges[:, 0], ranges[:, 1], size=(train_num, n_var)), dtype=tf.float32
    )
    test_input = tf.convert_to_tensor(
        np.random.uniform(ranges[:, 0], ranges[:, 1], size=(test_num, n_var)), dtype=tf.float32
    )

    if f_mode == 'col':
        train_label = f(train_input)
        test_label = f(test_input)
    elif f_mode == 'row':
        train_label = f(tf.transpose(train_input))
        test_label = f(tf.transpose(test_input))
    else:
        raise ValueError(f"Unrecognized f_mode: {f_mode}")

    if len(train_label.shape) == 1:
        train_label = tf.expand_dims(train_label, axis=-1)
        test_label = tf.expand_dims(test_label, axis=-1)

    if normalize_input:
        mean_input = tf.reduce_mean(train_input, axis=0, keepdims=True)
        std_input = tf.math.reduce_std(train_input, axis=0, keepdims=True)
        train_input = (train_input - mean_input) / std_input
        test_input = (test_input - mean_input) / std_input

    if normalize_label:
        mean_label = tf.reduce_mean(train_label, axis=0, keepdims=True)
        std_label = tf.math.reduce_std(train_label, axis=0, keepdims=True)
        train_label = (train_label - mean_label) / std_label
        test_label = (test_label - mean_label) / std_label

    return {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label,
    }
def batch_jacobian(func, x, create_graph=False, mode='scalar'):
    '''
    Compute Jacobian for a batch of inputs
    
    Args:
    -----
        func : function
            The function to compute the Jacobian for.
        x : tensor
            Input tensor of shape (Batch, Length).
        create_graph : bool
            Whether to create computation graph for further gradients.
        mode : str
            'scalar' or 'vector' Jacobian computation mode.
    
    Returns:
    --------
        jacobian : tensor
            Jacobian tensor.
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x)
    
    if mode == 'scalar':
        jacobian = tape.jacobian(y, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    elif mode == 'vector':
        jacobian = tf.stack([tape.jacobian(y_i, x) for y_i in tf.unstack(y, axis=1)], axis=1)
    else:
        raise ValueError("Invalid mode. Choose 'scalar' or 'vector'.")

    del tape
    return jacobian
def batch_hessian(func, x, create_graph=False):
    '''
    Compute Hessian for a batch of inputs
    
    Args:
    -----
        func : function
            The function to compute the Hessian for.
        x : tensor
            Input tensor of shape (Batch, Length).
        create_graph : bool
            Whether to create computation graph for further gradients.
    
    Returns:
    --------
        hessian : tensor
            Hessian tensor.
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x)
        jacobian = tape.jacobian(y, x)
    
    hessian = tape.jacobian(jacobian, x)
    del tape
    return hessian
def sparse_mask(in_dim, out_dim):
    '''
    Generate a sparse connection mask
    
    Args:
    -----
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
    
    Returns:
    --------
        mask : tensor
            A sparse mask tensor of shape (in_dim, out_dim).
    '''
    in_coord = tf.range(in_dim, dtype=tf.float32) * (1 / in_dim) + (1 / (2 * in_dim))
    out_coord = tf.range(out_dim, dtype=tf.float32) * (1 / out_dim) + (1 / (2 * out_dim))

    dist_mat = tf.abs(tf.expand_dims(out_coord, axis=1) - tf.expand_dims(in_coord, axis=0))
    in_nearest = tf.argmin(dist_mat, axis=0)
    out_nearest = tf.argmin(dist_mat, axis=1)

    mask = tf.zeros((in_dim, out_dim), dtype=tf.float32)
    for i, j in zip(range(in_dim), in_nearest.numpy()):
        mask = tf.tensor_scatter_nd_update(mask, [[j, i]], [1.0])
    for i, j in zip(out_nearest.numpy(), range(out_dim)):
        mask = tf.tensor_scatter_nd_update(mask, [[i, j]], [1.0])

    return mask
