import tensorflow as tf

import tensorflow as tf

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
    """
    # Expand dimensions for broadcasting
    if extend:
        x = tf.expand_dims(x, axis=-1)  # Shape: (batch_size, in_dim, 1)
        grid = tf.expand_dims(grid, axis=0)  # Shape: (1, in_dim, num_grid_points)

    if k == 0:
        # Base case: Zeroth-order B-spline
        value = tf.cast((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:]), tf.float32)
        return value
    else:
        # Recursive call for (k-1)-th B-spline
        B_km1 = B_batch(x[..., 0], grid[0], k=k - 1, extend=False)

        # Broadcast x and grid for current degree
        x = tf.expand_dims(x, axis=-1)  # Shape: (batch_size, in_dim, 1, 1)
        left_num = x - grid[:, :, :-k - 1]
        left_denom = grid[:, :, k:-1] - grid[:, :, :-k - 1] + 1e-8
        left = (left_num / left_denom) * B_km1[:, :, :-1]

        right_num = grid[:, :, k + 1:] - x
        right_denom = grid[:, :, k + 1:] - grid[:, :, 1:-k] + 1e-8
        right = (right_num / right_denom) * B_km1[:, :, 1:]

        # Combine left and right parts
        value = left + right

        # Handle NaN values
        value = tf.where(tf.math.is_nan(value), tf.zeros_like(value), value)
        return value




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
    b_splines = tf.transpose(b_splines, perm=[1, 0, 2])  # Transpose to match (in_dim, batch, grid_points + k)
    coef = tf.transpose(coef, perm=[0, 2, 1])  # Transpose coef to match (in_dim, grid_points + k, out_dim)

    # 确保 `einsum` 中维度对齐
    y_eval = tf.einsum('ijk,ikl->ijl', b_splines, coef)  # Shape: (in_dim, batch, out_dim)
    y_eval = tf.transpose(y_eval, perm=[1, 0, 2])  # Shape: (batch, in_dim, out_dim)
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
    mat_T = tf.transpose(mat, perm=[1, 0, 2])  # Shape: (in_dim, batch_size, num_basis_functions)
    y_eval_T = tf.transpose(y_eval, perm=[1, 0, 2])  # Shape: (in_dim, batch_size, out_dim)

    # Compute XtX and Xty for least squares
    XtX = tf.einsum('ijk,ilk->ijl', mat_T, mat_T)  # Shape: (in_dim, num_basis_functions, num_basis_functions)
    Xty = tf.einsum('ijk,ikl->ijl', mat_T, y_eval_T)  # Shape: (in_dim, num_basis_functions, out_dim)

    # Regularization
    reg = lamb * tf.eye(tf.shape(XtX)[-1], batch_shape=[tf.shape(XtX)[0]])
    coef = tf.linalg.solve(XtX + reg, Xty)  # Shape: (in_dim, num_basis_functions, out_dim)
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



