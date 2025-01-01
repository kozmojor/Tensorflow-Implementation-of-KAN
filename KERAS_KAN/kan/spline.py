import tensorflow as tf


def B_batch(x, grid, k=0, extend=True):
    """
    Evaluate x on B-spline bases using TensorFlow.

    Args:
    -----
        x : 2D tf.Tensor
            Inputs, shape (batch, in_dim)
        grid : 2D tf.Tensor
            Grids, shape (in_dim, grid_points)
        k : int
            The piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
    
    Returns:
    --------
        spline_values : 3D tf.Tensor
            Shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.

    Example:
    --------
    >>> import tensorflow as tf
    >>> x = tf.random.uniform((100, 2))
    >>> grid = tf.tile(tf.linspace(-1.0, 1.0, 11)[None, :], [2, 1])
    >>> B_values = B_batch(x, grid, k=3)
    >>> print(B_values.shape)
    """
    x = tf.expand_dims(x, axis=-1)  # Expand x to (batch, in_dim, 1)
    grid = tf.expand_dims(grid, axis=0)  # Expand grid to (1, in_dim, grid_points)

    if k == 0:
        # Compute spline basis for order k=0
        value = tf.cast((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:]), tf.float32)
    else:
        # Recursive computation for higher-order splines
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)

        # Compute left and right terms
        left_term = ((x - grid[:, :, :-(k + 1)]) / 
                     (grid[:, :, k:-1] - grid[:, :, :-(k + 1)])) * B_km1[:, :, :-1]
        right_term = ((grid[:, :, k + 1:] - x) / 
                      (grid[:, :, k + 1:] - grid[:, :, 1:-k])) * B_km1[:, :, 1:]

        value = left_term + right_term

    # Replace NaNs with zeros in case of degenerate grid
    value = tf.where(tf.math.is_nan(value), tf.zeros_like(value), value)
    return value


def coef2curve(x_eval, grid, coef, k):
    '''
    Convert B-spline coefficients to B-spline curves.

    Args:
    -----
        x_eval : 2D tf.Tensor
            shape (batch, in_dim)
        grid : 2D tf.Tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D tf.Tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
    
    Returns:
    --------
        y_eval : 3D tf.Tensor
            shape (batch, in_dim, out_dim)
    '''
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = tf.einsum('ijk,jlk->ijl', b_splines, coef)
    return y_eval


import tensorflow as tf

def curve2coef(x_eval, y_eval, grid, k):
    '''
    Converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D tf.Tensor
            shape (batch, in_dim)
        y_eval : 3D tf.Tensor
            shape (batch, in_dim, out_dim)
        grid : 2D tf.Tensor
            shape (in_dim, grid+2*k)
        k : int
            spline order

    Returns:
    --------
        coef : 3D tf.Tensor
            shape (in_dim, out_dim, G+k)
    '''
    batch = tf.shape(x_eval)[0]
    in_dim = tf.shape(x_eval)[1]
    out_dim = tf.shape(y_eval)[2]
    n_coef = tf.shape(grid)[1] - k - 1

    # Compute B-spline basis
    mat = B_batch(x_eval, grid, k)  # Use the TensorFlow version of B_batch
    mat = tf.transpose(mat, perm=[1, 0, 2])  # Permute dimensions
    mat = tf.expand_dims(mat, axis=1)  # Add dimension for output
    mat = tf.tile(mat, [1, out_dim, 1, 1])  # Expand for out_dim

    # Prepare y_eval for lstsq
    y_eval = tf.transpose(y_eval, perm=[1, 2, 0])
    y_eval = tf.expand_dims(y_eval, axis=3)

    # Solve least squares using TensorFlow
    try:
        coef = tf.linalg.lstsq(mat, y_eval)[:, :, :, 0]  # Solve least squares and remove last dim
    except Exception as e:
        print("lstsq failed:", e)
        raise

    return coef


def extend_grid(grid, k_extend=0):
    '''
    Extend grid.

    Args:
    -----
        grid : 2D tf.Tensor
            Grids to be extended.
        k_extend : int
            Number of extensions on both sides.
    
    Returns:
    --------
        Extended grid : 2D tf.Tensor
    '''
    h = (grid[:, -1:] - grid[:, :1]) / tf.cast((grid.shape[1] - 1), dtype=grid.dtype)

    for i in range(k_extend):
        grid = tf.concat([grid[:, :1] - h, grid], axis=1)
        grid = tf.concat([grid, grid[:, -1:] + h], axis=1)

    return tf.cast(grid, dtype=tf.float32)

