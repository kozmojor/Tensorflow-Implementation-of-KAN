import tensorflow as tf
import numpy as np
from sklearn.linear_model import LinearRegression
import sympy
import yaml
from sympy.utilities.lambdify import lambdify
import re

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

from sympy import lambdify

def augment_input(orig_vars, aux_vars, x):
    """
    Augment input features with additional symbolic transformations.

    Args:
        orig_vars: list of sympy symbols, representing original variables.
        aux_vars: list of sympy expressions, auxiliary variables to compute.
        x: tf.Tensor, input tensor of shape (num_samples, num_features).

    Returns:
        Augmented input tensor including auxiliary variables.

    Example:
    >>> from sympy import symbols
    >>> orig_vars = symbols('a b')
    >>> aux_vars = [orig_vars[0] + orig_vars[1], orig_vars[0] * orig_vars[1]]
    >>> x = tf.random.normal((100, 2))
    >>> augmented_x = augment_input(orig_vars, aux_vars, x)
    >>> augmented_x.shape
    TensorShape([100, 4])
    """
    if isinstance(x, dict):  # For dataset augmentation
        x['train_input'] = augment_input(orig_vars, aux_vars, x['train_input'])
        x['test_input'] = augment_input(orig_vars, aux_vars, x['test_input'])
        return x

    # For direct input tensor
    aux_values = []
    for aux_var in aux_vars:
        func = lambdify(orig_vars, aux_var, 'numpy')  # SymPy to NumPy conversion
        aux_values.append(tf.convert_to_tensor(func(*[x[:, i].numpy() for i in range(len(orig_vars))])))
    aux_values = tf.stack(aux_values, axis=1)

    return tf.concat([x, aux_values], axis=1)


def sparse_mask(in_dim, out_dim):
    """
    Generate a sparse connectivity mask.

    Args:
        in_dim: int
            Input dimension.
        out_dim: int
            Output dimension.

    Returns:
        A sparse mask tensor of shape (in_dim, out_dim).
    """
    in_coord = tf.range(in_dim) / in_dim + 1 / (2 * in_dim)
    out_coord = tf.range(out_dim) / out_dim + 1 / (2 * out_dim)

    dist_mat = tf.abs(tf.expand_dims(out_coord, -1) - tf.expand_dims(in_coord, 0))
    in_nearest = tf.argmin(dist_mat, axis=0)
    out_nearest = tf.argmin(dist_mat, axis=1)

    mask = tf.zeros((in_dim, out_dim), dtype=tf.float64)
    mask = tf.tensor_scatter_nd_update(mask, tf.stack([tf.range(in_dim, dtype=tf.int64), in_nearest], axis=1), tf.ones(in_dim, dtype=mask.dtype))
    mask = tf.tensor_scatter_nd_update(mask, tf.stack([out_nearest, tf.range(out_dim, dtype=tf.int64)], axis=1), tf.ones(out_dim, dtype=mask.dtype))

    return mask

f_inv = lambda x, y_th: (
    (x_th := 1 / y_th),
    y_th / x_th * x * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / x, tf.zeros_like(x))
)

f_inv2 = lambda x, y_th: (
    (x_th := 1 / y_th**(1 / 2)),
    y_th * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / x**2, tf.zeros_like(x))
)

f_inv3 = lambda x, y_th: (
    (x_th := 1 / y_th**(1 / 3)),
    y_th / x_th * x * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / x**3, tf.zeros_like(x))
)

f_inv4 = lambda x, y_th: (
    (x_th := 1 / y_th**(1 / 4)),
    y_th * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / x**4, tf.zeros_like(x))
)

f_inv5 = lambda x, y_th: (
    (x_th := 1 / y_th**(1 / 5)),
    y_th / x_th * x * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / x**5, tf.zeros_like(x))
)

f_sqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    x_th / y_th * x * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, tf.sqrt(tf.abs(x)) * tf.sign(x), tf.zeros_like(x))
)

f_power1d5 = lambda x, y_th: tf.abs(x)**1.5

f_invsqrt = lambda x, y_th: (
    (x_th := 1 / y_th**2),
    y_th * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, 1 / tf.sqrt(tf.abs(x)), tf.zeros_like(x))
)

f_log = lambda x, y_th: (
    (x_th := tf.exp(-y_th)),
    -y_th * (tf.abs(x) < x_th) + tf.where(tf.abs(x) >= x_th, tf.math.log(tf.abs(x)), tf.zeros_like(x))
)

f_tan = lambda x, y_th: (
    (clip := x % tf.constant(tf.experimental.numpy.pi)),
    (delta := tf.constant(tf.experimental.numpy.pi) / 2 - tf.atan(y_th)),
    -y_th / delta * (clip - tf.constant(tf.experimental.numpy.pi) / 2) * (tf.abs(clip - tf.constant(tf.experimental.numpy.pi) / 2) < delta) +
    tf.where(tf.abs(clip - tf.constant(tf.experimental.numpy.pi) / 2) >= delta, tf.tan(clip), tf.zeros_like(x))
)

f_arctanh = lambda x, y_th: (
    (delta := 1 - tf.tanh(y_th) + 1e-4),
    y_th * tf.sign(x) * (tf.abs(x) > 1 - delta) +
    tf.where(tf.abs(x) <= 1 - delta, tf.atanh(x), tf.zeros_like(x))
)

f_arcsin = lambda x, y_th: (
    (),
    tf.constant(tf.experimental.numpy.pi) / 2 * tf.sign(x) * (tf.abs(x) > 1) +
    tf.where(tf.abs(x) <= 1, tf.asin(x), tf.zeros_like(x))
)

f_arccos = lambda x, y_th: (
    (),
    tf.constant(tf.experimental.numpy.pi) / 2 * (1 - tf.sign(x)) * (tf.abs(x) > 1) +
    tf.where(tf.abs(x) <= 1, tf.acos(x), tf.zeros_like(x))
)

f_exp = lambda x, y_th: (
    (x_th := tf.math.log(y_th)),
    y_th * (x > x_th) + tf.exp(x) * (x <= x_th)
)


SYMBOLIC_LIB = {
    'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
    'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
    'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
    'x^4': (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
    'x^5': (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
    '1/x': (lambda x: 1/x, lambda x: 1/x, 2, f_inv),
    '1/x^2': (lambda x: 1/x**2, lambda x: 1/x**2, 2, f_inv2),
    '1/x^3': (lambda x: 1/x**3, lambda x: 1/x**3, 3, f_inv3),
    '1/x^4': (lambda x: 1/x**4, lambda x: 1/x**4, 4, f_inv4),
    '1/x^5': (lambda x: 1/x**5, lambda x: 1/x**5, 5, f_inv5),
    'sqrt': (lambda x: tf.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    'x^0.5': (lambda x: tf.sqrt(x), lambda x: sympy.sqrt(x), 2, f_sqrt),
    'x^1.5': (lambda x: tf.sqrt(x)**3, lambda x: sympy.sqrt(x)**3, 4, f_power1d5),
    '1/sqrt(x)': (lambda x: 1/tf.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
    '1/x^0.5': (lambda x: 1/tf.sqrt(x), lambda x: 1/sympy.sqrt(x), 2, f_invsqrt),
    'exp': (lambda x: tf.exp(x), lambda x: sympy.exp(x), 2, f_exp),
    'log': (lambda x: tf.math.log(x), lambda x: sympy.log(x), 2, f_log),
    'abs': (lambda x: tf.abs(x), lambda x: sympy.Abs(x), 3, lambda x, y_th: ((), tf.abs(x))),
    'sin': (lambda x: tf.sin(x), lambda x: sympy.sin(x), 2, lambda x, y_th: ((), tf.sin(x))),
    'cos': (lambda x: tf.cos(x), lambda x: sympy.cos(x), 2, lambda x, y_th: ((), tf.cos(x))),
    'tan': (lambda x: tf.tan(x), lambda x: sympy.tan(x), 3, f_tan),
    'tanh': (lambda x: tf.tanh(x), lambda x: sympy.tanh(x), 3, lambda x, y_th: ((), tf.tanh(x))),
    'sgn': (lambda x: tf.sign(x), lambda x: sympy.sign(x), 3, lambda x, y_th: ((), tf.sign(x))),
    'arcsin': (lambda x: tf.asin(x), lambda x: sympy.asin(x), 4, f_arcsin),
    'arccos': (lambda x: tf.acos(x), lambda x: sympy.acos(x), 4, f_arccos),
    'arctan': (lambda x: tf.atan(x), lambda x: sympy.atan(x), 4, lambda x, y_th: ((), tf.atan(x))),
    'arctanh': (lambda x: tf.atanh(x), lambda x: sympy.atanh(x), 4, f_arctanh),
    '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
    'gaussian': (lambda x: tf.exp(-x**2), lambda x: sympy.exp(-x**2), 3, lambda x, y_th: ((), tf.exp(-x**2))),
}



def batch_jacobian(func, x, mode='scalar'):
    """
    Compute the Jacobian for a batch of inputs using TensorFlow.

    Args:
        func: Callable function or model, expected to process inputs `x`.
        x: tf.Tensor, inputs for which Jacobian is computed.
        mode: str, either 'scalar' (scalar function) or 'vector' (vector-valued function).

    Returns:
        jacobian: tf.Tensor, Jacobian matrix of shape (batch_size, output_dim, input_dim).

    Example:
    >>> model = lambda x: x[:, :1] ** 2 + x[:, 1:]
    >>> x = tf.random.normal((10, 3))
    >>> jacobian = batch_jacobian(model, x)
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = func(x)
    if mode == 'scalar':
        jacobian = tape.jacobian(y, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    elif mode == 'vector':
        jacobian = tf.stack([tape.jacobian(y_i, x) for y_i in tf.unstack(y, axis=-1)], axis=-1)
    else:
        raise ValueError(f"Mode {mode} not recognized.")
    del tape
    return jacobian

def batch_hessian(model, x):
    """
    Compute the Hessian for a batch of inputs using TensorFlow.

    Args:
        model: Callable function or model.
        x: tf.Tensor, inputs for which Hessian is computed.

    Returns:
        hessian: tf.Tensor, Hessian tensor of shape (batch_size, input_dim, input_dim).

    Example:
    >>> model = lambda x: tf.reduce_sum(x[:, :1] ** 2 + x[:, 1:] ** 3, axis=1)
    >>> x = tf.random.normal((10, 3))
    >>> hessian = batch_hessian(model, x)
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = model(x)
        grad = tape.gradient(y, x)
    hessian = tf.stack([tape.jacobian(g, x) for g in tf.unstack(grad, axis=-1)], axis=-1)
    del tape
    return hessian

def get_derivative(model, inputs, labels, derivative='hessian', loss_mode='pred', lamb=0.0):
    def loss_fn():
        preds = model(inputs)
        pred_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(labels, preds))
        reg_loss = tf.add_n([tf.nn.l2_loss(w) for w in model.trainable_weights])
        if loss_mode == 'pred':
            return pred_loss
        elif loss_mode == 'reg':
            return lamb * reg_loss
        elif loss_mode == 'all':
            return pred_loss + lamb * reg_loss
        else:
            raise ValueError(f"Loss mode {loss_mode} not recognized.")

    if derivative == 'jacobian':
        with tf.GradientTape() as tape:
            loss = loss_fn()
        gradients = tape.jacobian(loss, model.trainable_weights)
        return gradients

    elif derivative == 'hessian':
        with tf.GradientTape(persistent=True) as tape:
            loss = loss_fn()
            gradients = tape.gradient(loss, model.trainable_weights)
        hessians = [tape.jacobian(g, model.trainable_weights) for g in gradients]
        del tape
        return hessians

    else:
        raise ValueError(f"Derivative type {derivative} not recognized.")



def create_dataset_from_data(inputs, labels, train_ratio=0.8):
    """
    Create a TensorFlow-compatible dataset from raw data.

    Args:
        inputs: np.ndarray or tf.Tensor, input features.
        labels: np.ndarray or tf.Tensor, target labels.
        train_ratio: float, the ratio of training data to the total data.

    Returns:
        dataset: dict containing 'train_input', 'test_input', 'train_label', 'test_label'.

    Example:
    >>> inputs = tf.random.normal((100, 5))
    >>> labels = tf.random.normal((100, 1))
    >>> dataset = create_dataset_from_data(inputs, labels)
    >>> dataset['train_input'].shape
    TensorShape([80, 5])
    """
    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(num_samples * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    dataset = {
        'train_input': tf.convert_to_tensor(inputs[train_indices]),
        'test_input': tf.convert_to_tensor(inputs[test_indices]),
        'train_label': tf.convert_to_tensor(labels[train_indices]),
        'test_label': tf.convert_to_tensor(labels[test_indices]),
    }
    return dataset
def fit_params(x, y, fun, a_range=(-10, 10), b_range=(-10, 10), grid_number=101, iteration=3, verbose=True):
    """
    Fit a, b, c, d such that:
        |y - (c * fun(a * x + b) + d)|^2
    is minimized. Both x and y are 1D tensors.

    Args:
    -----
        x : tf.Tensor
            1D tensor for x values.
        y : tf.Tensor
            1D tensor for y values.
        fun : function
            Symbolic function.
        a_range : tuple
            Sweeping range of a.
        b_range : tuple
            Sweeping range of b.
        grid_number : int
            Number of steps along a and b.
        iteration : int
            Number of zooming iterations.
        verbose : bool
            Print extra information if True.

    Returns:
    --------
        a_best : tf.Tensor
            Best fitted a.
        b_best : tf.Tensor
            Best fitted b.
        c_best : tf.Tensor
            Best fitted c.
        d_best : tf.Tensor
            Best fitted d.
        r2_best : tf.Tensor
            Best R^2 (coefficient of determination).
    """
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    for _ in range(iteration):
        a_ = tf.cast(tf.linspace(a_range[0], a_range[1], grid_number), tf.float32)
        b_ = tf.cast(tf.linspace(b_range[0], b_range[1], grid_number), tf.float32)
        a_grid, b_grid = tf.meshgrid(a_, b_, indexing='ij')

        post_fun = fun(a_grid[None, :, :] * x[:, None, None] + b_grid[None, :, :])

        x_mean = tf.reduce_mean(post_fun, axis=0, keepdims=True)
        y_mean = tf.reduce_mean(y, keepdims=True)
        numerator = tf.reduce_sum((post_fun - x_mean) * (y[:, None, None] - y_mean), axis=0) ** 2
        denominator = tf.reduce_sum((post_fun - x_mean) ** 2, axis=0) * tf.reduce_sum((y - y_mean) ** 2)

        r2 = numerator / (denominator + 1e-4)
        r2 = tf.where(tf.math.is_nan(r2), tf.zeros_like(r2), r2)

        best_id = tf.argmax(tf.reshape(r2, [-1]))
        a_id = best_id // grid_number
        b_id = best_id % grid_number

        if a_id == 0 or a_id == grid_number - 1 or b_id == 0 or b_id == grid_number - 1:
            if _ == 0 and verbose:
                print("Best value at boundary.")
            if a_id == 0:
                a_range = [a_[0], a_[1]]
            if a_id == grid_number - 1:
                a_range = [a_[-2], a_[-1]]
            if b_id == 0:
                b_range = [b_[0], b_[1]]
            if b_id == grid_number - 1:
                b_range = [b_[-2], b_[-1]]
        else:
            a_range = [a_[a_id - 1], a_[a_id + 1]]
            b_range = [b_[b_id - 1], b_[b_id + 1]]

    a_best = a_[a_id]
    b_best = b_[b_id]
    post_fun = fun(a_best * x + b_best)
    r2_best = r2[a_id, b_id]

    if verbose:
        print(f"r2 is {r2_best.numpy()}")
        if r2_best < 0.9:
            print("r2 is not very high, please double-check if you are choosing the correct symbolic function.")

    post_fun = tf.where(tf.math.is_nan(post_fun), tf.zeros_like(post_fun), post_fun)
    reg = LinearRegression().fit(post_fun[:, None].numpy(), y.numpy())
    c_best = tf.constant(reg.coef_[0], dtype=tf.float32)
    d_best = tf.constant(reg.intercept_, dtype=tf.float32)

    return tf.stack([a_best, b_best, c_best, d_best]), r2_best

def add_symbolic(name, func, c=1, func_singularity=None):
    """
    Dynamically add symbolic functions to the SYMBOLIC_LIB for compatibility with symbolic computations.

    Args:
        name: str, name of the function.
        func: Callable, TensorFlow function implementation.
        c: int, complexity weight.
        func_singularity: Callable, singularity-protected version of the function (optional).

    Example:
    >>> add_symbolic('Bessel', tf.math.bessel_j0)
    >>> SYMBOLIC_LIB['Bessel']
    (<function bessel_j0 at 0x...>, Bessel)
    """
    globals()[name] = Function(name)
    if func_singularity is None:
        func_singularity = func
    SYMBOLIC_LIB[name] = (func, globals()[name], c, func_singularity)

from sympy import Float

def ex_round(expr, n_digit):
    """
    Round the floating-point numbers in a SymPy expression to a specific number of digits.

    Args:
        expr: sympy.Expr, the input expression.
        n_digit: int, number of decimal digits to round to.

    Returns:
        sympy.Expr, the rounded expression.

    Example:
    >>> from sympy import symbols, exp, sin
    >>> x = symbols('x')
    >>> expr = 3.14534242 * exp(sin(x)) - 2.32345402
    >>> ex_round(expr, 2)
    3.15*exp(sin(x)) - 2.32
    """
    for sub_expr in expr.atoms(Float):
        expr = expr.subs(sub_expr, round(sub_expr, n_digit))
    return expr

def model2param(model):
    params = []
    for var in model.trainable_variables:
        params.append(tf.reshape(var, [-1]))
    return tf.concat(params, axis=0)

def param2statedict(flat_params, model):

    start = 0
    state_dict = {}
    for var in model.trainable_variables:
        num_params = tf.size(var).numpy()
        reshaped_params = tf.reshape(flat_params[start : start + num_params], var.shape)
        state_dict[var.name] = reshaped_params
        start += num_params
    return state_dict



