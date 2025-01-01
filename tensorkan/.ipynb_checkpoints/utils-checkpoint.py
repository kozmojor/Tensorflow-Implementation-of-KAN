import tensorflow as tf
import numpy as np

def create_dataset(f, n_var=2, f_mode='col', ranges=[-1, 1], train_num=1000, test_num=1000, normalize_input=False, normalize_label=False, seed=0):
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

    mask = tf.zeros((in_dim, out_dim), dtype=tf.float32)
    mask = tf.tensor_scatter_nd_update(mask, tf.stack([tf.range(in_dim), in_nearest], axis=1), tf.ones(in_dim))
    mask = tf.tensor_scatter_nd_update(mask, tf.stack([out_nearest, tf.range(out_dim)], axis=1), tf.ones(out_dim))

    return mask

# Not fully transplanted yet below

def f_inv(x, y_th):
    x_th = 1 / y_th
    return tf.where(tf.abs(x) < x_th, y_th / x_th * x, tf.math.divide_no_nan(1.0, x))

def f_inv2(x, y_th):
    x_th = 1 / tf.sqrt(y_th)
    return tf.where(tf.abs(x) < x_th, y_th, tf.math.divide_no_nan(1.0, tf.pow(x, 2)))

def f_inv3(x, y_th):
    x_th = 1 / tf.pow(y_th, 1/3)
    return tf.where(tf.abs(x) < x_th, y_th / x_th * x, tf.math.divide_no_nan(1.0, tf.pow(x, 3)))

def f_inv4(x, y_th):
    x_th = 1 / tf.pow(y_th, 1/4)
    return tf.where(tf.abs(x) < x_th, y_th, tf.math.divide_no_nan(1.0, tf.pow(x, 4)))

def f_log(x, y_th):
    x_th = tf.exp(-y_th)
    return tf.where(tf.abs(x) < x_th, -y_th, tf.math.log(tf.abs(x)))

def f_exp(x, y_th):
    x_th = tf.math.log(y_th)
    return tf.where(x > x_th, y_th, tf.exp(x))

def f_sqrt(x, y_th):
    x_th = 1 / (y_th ** 2)
    return tf.where(
        tf.abs(x) < x_th,
        x_th / y_th * x,
        tf.sqrt(tf.abs(x)) * tf.sign(x),
    )

SYMBOLIC_LIB = {
    'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: x),
    'x^2': (lambda x: tf.pow(x, 2), lambda x: tf.pow(x, 2), 2, lambda x, y_th: tf.pow(x, 2)),
    'x^3': (lambda x: tf.pow(x, 3), lambda x: tf.pow(x, 3), 3, lambda x, y_th: tf.pow(x, 3)),
    'x^4': (lambda x: tf.pow(x, 4), lambda x: tf.pow(x, 4), 4, lambda x, y_th: tf.pow(x, 4)),
    '1/x': (lambda x: tf.math.divide_no_nan(1.0, x), lambda x: 1 / x, 2, f_inv),
    '1/x^2': (lambda x: tf.math.divide_no_nan(1.0, tf.pow(x, 2)), lambda x: 1 / tf.pow(x, 2), 2, f_inv2),
    'sqrt': (lambda x: tf.sqrt(x), lambda x: tf.sqrt(x), 2, f_sqrt),
    'exp': (tf.exp, lambda x: tf.exp(x), 2, f_exp),
    'log': (tf.math.log, lambda x: tf.math.log(x), 2, f_log),
    'sin': (tf.sin, lambda x: tf.sin(x), 2, lambda x, y_th: tf.sin(x)),
    'cos': (tf.cos, lambda x: tf.cos(x), 2, lambda x, y_th: tf.cos(x)),
    'tan': (tf.tan, lambda x: tf.tan(x), 3, lambda x, y_th: tf.tan(x)),
    'tanh': (tf.tanh, lambda x: tf.tanh(x), 3, lambda x, y_th: tf.tanh(x)),
    'arcsin': (tf.asin, lambda x: tf.asin(x), 4, lambda x, y_th: tf.asin(x)),
    'arccos': (tf.acos, lambda x: tf.acos(x), 4, lambda x, y_th: tf.acos(x)),
    'arctanh': (tf.atanh, lambda x: tf.atanh(x), 4, f_inv),
    '0': (lambda x: tf.zeros_like(x), lambda x: 0 * x, 0, lambda x, y_th: tf.zeros_like(x)),
}

import tensorflow as tf

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

def fit_params(x, y, func, a_range=(-10, 10), b_range=(-10, 10), grid_number=101, iteration=3, verbose=True):
    a_grid = tf.linspace(a_range[0], a_range[1], grid_number)
    b_grid = tf.linspace(b_range[0], b_range[1], grid_number)

    best_r2 = -float('inf')
    best_params = None

    for _ in range(iteration):
        r2_scores = []
        for a in a_grid:
            for b in b_grid:
                predictions = func(a * x + b)
                residuals = y - predictions
                r2 = 1 - tf.reduce_sum(residuals**2) / tf.reduce_sum((y - tf.reduce_mean(y))**2)
                r2_scores.append((r2.numpy(), a.numpy(), b.numpy()))

        best_r2, best_a, best_b = max(r2_scores, key=lambda x: x[0])

        # Refinement of search range
        a_range = (best_a - (a_range[1] - a_range[0]) / grid_number, best_a + (a_range[1] - a_range[0]) / grid_number)
        b_range = (best_b - (b_range[1] - b_range[0]) / grid_number, best_b + (b_range[1] - b_range[0]) / grid_number)
        a_grid = tf.linspace(a_range[0], a_range[1], grid_number)
        b_grid = tf.linspace(b_range[0], b_range[1], grid_number)

    predictions = func(best_a * x + best_b)
    A = tf.stack([predictions, tf.ones_like(predictions)], axis=1)
    c, d = tf.linalg.lstsq(A, tf.expand_dims(y, axis=-1), l2_regularizer=1e-6)

    if verbose:
        print(f"Best R²: {best_r2}")

    return tf.convert_to_tensor([best_a, best_b, c[0][0], d[0][0]]), best_r2


from sympy import Function
import tensorflow as tf

SYMBOLIC_LIB = {}  # 使用动态注册的符号库

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



