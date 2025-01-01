import tensorflow as tf


    
def B_batch(x, grid, k=0, extend=True):
    """
    Evaluate x on B-spline basis functions.

    Args:
        x: tf.Tensor, shape (batch_size, in_dim).
        grid: tf.Tensor, shape (in_dim, num_grid_points).
        k: int, B-spline degree.
        extend: bool, if True, extend the basis functions.

    Returns:
        B-spline values: tf.Tensor, shape (batch_size, in_dim, num_basis_functions).
        
    Example:
    --------
    >>> import tensorflow as tf
    >>> x = tf.random.uniform((100, 2))
    >>> grid = tf.tile(tf.linspace(-1.0, 1.0, 11)[None, :], [2, 1])
    >>> B_values = B_batch(x, grid, k=3)
    >>> print(B_values.shape)
    """
    # Expand dimensions for broadcasting
    if extend:
        x = tf.expand_dims(x, axis=-1)  
       
    
    # Initialize order-0 B-spline bases
    bases = tf.logical_and(
        tf.greater_equal(x, grid[:, :-1]),
        tf.less(x, grid[:, 1:])
    )
    bases = tf.cast(bases, dtype=tf.float32)
    
    # Iteratively calculate higher-order B-spline bases
    for order in range(1, k + 1):
        bases = (
            (x - grid[:, : -(order + 1)]) / (grid[:, order:-1] - grid[:, : -(order + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, order + 1 :] - x) / (grid[:, order + 1 :] - grid[:, 1:(-order)])
            * bases[:, :, 1:]
        )

    # Handle NaN values (in case of invalid divisions)
    bases = tf.where(tf.math.is_nan(bases), tf.zeros_like(bases), bases)

    return bases



def coef2curve(x_eval, grid, coef, k):
    """
    Convert B-spline coefficients to curves. Evaluate x on B-spline curves.

    Args:
        x_eval: 2D tensor, shape (batch, in_dim)
        grid: 2D tensor, shape (in_dim, grid_points + 2k)
        coef: 3D tensor, shape (in_dim, out_dim, grid_points + k)
        k: int, piecewise polynomial order of splines

    Returns:
        y_eval: 3D tensor, shape (batch, in_dim, out_dim)
    """
    b_splines = B_batch(x_eval, grid, k=k)  # Shape: (batch, in_dim, grid_points + k)
    # No need for transpose if alignment matches torch version
    y_eval = tf.einsum('ijk,jlk->ijl', b_splines, coef)  # Align einsum formula with torch version
    return y_eval

def curve2coef(x_eval, y_eval, grid, k, lamb=1e-8):
    """
    Convert B-spline curves to coefficients using least squares.

    Args:
        x_eval: tf.Tensor, shape (batch_size, in_dim).
        y_eval: tf.Tensor, shape (batch_size, in_dim, out_dim).
        grid: tf.Tensor, shape (in_dim, num_grid_points).
        k: int, spline degree.
        lamb: float, regularization parameter.

    Returns:
        coef: tf.Tensor, shape (in_dim, out_dim, num_basis_functions).
    """
    # Evaluate B-spline basis
    mat = B_batch(x_eval, grid, k=k)  # Shape: (batch_size, in_dim, num_basis_functions)

    # Transpose for batch-wise least squares
    mat_T = tf.cast(tf.transpose(mat, perm=[1, 0, 2]), dtype=tf.float32)
    y_eval_T = tf.cast(tf.transpose(y_eval, perm=[1, 2, 0]), dtype=tf.float32)  # Shape: (in_dim, out_dim, batch_size)
    y_eval_T = tf.expand_dims(y_eval_T, axis=3)  # Shape: (in_dim, out_dim, batch_size, 1)
    
    mat_T = tf.expand_dims(mat_T, axis=1)  # Shape: (in_dim, 1, batch_size, num_basis_functions)
    mat_T = tf.tile(mat_T, [1, y_eval.shape[2], 1, 1])  # Shape: (in_dim, out_dim, batch_size, num_basis_functions)


    # Compute XtX and Xty for least squares
    XtX = tf.einsum('ijmn,ijnp->ijmp', tf.transpose(mat_T, perm=[0, 1, 3, 2]), mat_T)  # Shape: (in_dim, num_basis_functions, num_basis_functions)
    Xty = tf.einsum('ijmn,ijnp->ijmp', tf.transpose(mat_T, perm=[0, 1, 3, 2]), y_eval_T)  # Shape: (in_dim, num_basis_functions, out_dim)

    # Regularization
    # Compute identity and regularization
    n = tf.shape(XtX)[-1]
    identity = tf.eye(n, dtype=tf.float32)  # Shape: (num_basis_functions, num_basis_functions)
    identity = tf.expand_dims(identity, axis=0)  # Shape: (1, num_basis_functions, num_basis_functions)
    identity = tf.expand_dims(identity, axis=0)  # Shape: (1, 1, num_basis_functions, num_basis_functions)
    identity = tf.tile(identity, [tf.shape(XtX)[0], tf.shape(XtX)[1], 1, 1])  # Shape: (in_dim, out_dim, num_basis_functions, num_basis_functions)

    reg = lamb * identity
    A = XtX + reg

    # Solve for coefficients
    B = Xty  # Shape: (in_dim, out_dim, num_basis_functions, num_basis_functions)
    coef = tf.linalg.solve(A, B)  # Shape: (in_dim, out_dim, num_basis_functions)
    coef = tf.squeeze(coef, axis=-1)  # Remove the last dimension, final shape: (batch, in_dim, num_basis_functions)
    return coef



def extend_grid(grid, k_extend=0):
    """
    Extend grid with additional points.

    Args:
        grid: tf.Tensor, shape (in_dim, num_grid_points).
        k_extend: int, number of additional points to extend.

    Returns:
        Extended grid: tf.Tensor, shape (in_dim, num_grid_points + 2*k_extend).
    """
    # Compute step size
    h = (grid[:, -1:] - grid[:, :1]) / tf.cast(tf.shape(grid)[1] - 1, grid.dtype)

    # Extend on both ends
    for _ in range(k_extend):
        grid = tf.concat([grid[:, :1] - h, grid], axis=1)  # Extend left
        grid = tf.concat([grid, grid[:, -1:] + h], axis=1)  # Extend right
    return grid



