import tensorflow as tf

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    """
        Perform cubic interpolation to determine the step size in line search.

        Args:
            x1, x2: float
                Two positions (previous step and current step).
            f1, f2: float
                Function values at x1 and x2.
            g1, g2: float
                Gradients at x1 and x2.
            bounds: tuple, optional
                Bounds for the step size.

        Returns:
            min_pos: float
                The interpolated position that minimizes the function.
        """
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2) # Intermediate term for interpolation
    d2_square = d1**2 - g1 * g2 # Discriminant to check for valid interpolation

    if d2_square >= 0:
        d2 = tf.sqrt(d2_square)
        # Update step size based on cubic interpolation formula
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return tf.clip_by_value(min_pos, xmin_bound, xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0 # If interpolation fails, return midpoint


def _strong_wolfe(obj_func,
                  x,
                  t,
                  d,
                  f,
                  g,
                  gtd,
                  c1=1e-4,
                  c2=0.9,
                  tolerance_change=1e-9,
                  max_ls=25):
    """
    Perform line search using the strong Wolfe conditions.

    Args:
        obj_func: Callable
            Function to evaluate objective and gradient.
        x: Tensor
            Current position.
        t: float
            Initial step size.
        d: Tensor
            Search direction.
        f: float
            Current function value.
        g: Tensor
            Current gradient.
        gtd: float
            Directional derivative at the current position.
        c1, c2: float
            Parameters for the strong Wolfe conditions.
        tolerance_change: float
            Convergence tolerance for step size changes.
        max_ls: int
            Maximum number of line search iterations.

    Returns:
        f_new: float
            New function value.
        g_new: Tensor
            New gradient.
        t: float
            Final step size.
        ls_func_evals: int
            Number of function evaluations during line search.
    """
    d_norm = tf.reduce_max(tf.abs(d))
    g = tf.identity(g)

    f_new, g_new = obj_func(x, t, d) # Evaluate function at initial step
    ls_func_evals = 1
    gtd_new = tf.tensordot(g_new, d, axes=1) # Compute directional derivative

    t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g, gtd
    done = False
    ls_iter = 0

    while ls_iter < max_ls: # Begin line search iterations
        # Check first Wolfe condition: sufficient decrease
        if (f_new > f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, tf.identity(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # Check second Wolfe condition: curvature condition
        if tf.abs(gtd_new) <= -c2 * gtd:
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        # If gradient turns positive, a minimum is bracketed
        if gtd_new >= 0:
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, tf.identity(g_new)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # Extrapolate step size using cubic interpolation
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10.0
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))

        t_prev = tmp
        f_prev = f_new
        g_prev = tf.identity(g_new)
        gtd_prev = gtd_new

        f_new, g_new = obj_func(x, t, d) # Evaluate new function value and gradient
        ls_func_evals += 1
        gtd_new = tf.tensordot(g_new, d, axes=1)
        ls_iter += 1

    if ls_iter == max_ls:
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]
        bracket_gtd = [gtd, gtd_new]

    # Bisection or interpolation to refine step size in the bracket
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    insuf_progress = False

    while not done and ls_iter < max_ls:
        if tf.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break

        t = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                               bracket[1], bracket_f[1], bracket_gtd[1])

        # Prevent insufficient progress
        eps = 0.1 * (tf.reduce_max(bracket) - tf.reduce_min(bracket))
        if tf.reduce_min([tf.reduce_max(bracket) - t, t - tf.reduce_min(bracket)]) < eps:
            if insuf_progress or t >= tf.reduce_max(bracket) or t <= tf.reduce_min(bracket):
                if tf.abs(t - tf.reduce_max(bracket)) < tf.abs(t - tf.reduce_min(bracket)):
                    t = tf.reduce_max(bracket) - eps
                else:
                    t = tf.reduce_min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        f_new, g_new = obj_func(x, t, d) # Evaluate at new step size
        ls_func_evals += 1
        gtd_new = tf.tensordot(g_new, d, axes=1)
        ls_iter += 1

        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = tf.identity(g_new)
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if tf.abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = tf.identity(g_new)
            bracket_gtd[low_pos] = gtd_new

    if len(bracket) == 1:
        t = bracket[0]
        f_new = bracket_f[0]
        g_new = bracket_g[0]
    else:
        t = bracket[low_pos]
        f_new = bracket_f[low_pos]
        g_new = bracket_g[low_pos]

    return f_new, g_new, t, ls_func_evals


class LBFGS(tf.keras.optimizers.Optimizer):
    """
        Implements the Limited-memory BFGS (L-BFGS) optimization algorithm in TensorFlow.

        Key Features:
            - Combines gradient information from past iterations for efficient search direction.
            - Supports Strong Wolfe line search for determining optimal step size.
            - Efficient memory usage for large-scale optimization problems.

        Args:
            learning_rate: float
                Base step size for updates.
            max_iter: int
                Maximum iterations for optimization.
            history_size: int
                Number of previous gradients and steps to store for direction calculation.
            tolerance_grad, tolerance_change: float
                Convergence criteria for gradients and step size changes.
            line_search_fn: str
                Line search method, defaults to 'strong_wolfe'.
    """
    def __init__(self,
                 learning_rate=0.0001,
                 max_iter=20,
                 max_eval=None,
                 tolerance_grad=1e-7,
                 tolerance_change=1e-9,
                 tolerance_ys=1e-32,
                 history_size=100,
                 line_search_fn='strong_wolfe',
                 name="LBFGS",
                 **kwargs):
        # Initialize the base TensorFlow optimizer
        super().__init__(name, **kwargs)

        # Set parameters for the optimizer
        self._learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_eval = max_eval if max_eval is not None else int(max_iter * 1.25)
        self.tolerance_grad = tolerance_grad
        self.tolerance_change = tolerance_change
        self.tolerance_ys = tolerance_ys
        self.history_size = history_size
        self.line_search_fn = line_search_fn

        # Internal state for optimization
        self.state = {
            'func_evals': 0, # Number of function evaluations
            'n_iter': 0,  # Current iteration count
            'old_dirs': [], # Stored update directions
            'old_stps': [], # Stored gradient differences
            'ro': [], # Scaling factors for updates
            'H_diag': 1.0, # Diagonal scaling factor for Hessian approximation
            'prev_flat_grad': None, # Flattened gradient at previous step
            'prev_loss': None, # Loss at previous step
            'direction': None, # Search direction
            't': None # Step size
        }

    @property
    def learning_rate(self):
        """Return the learning rate."""
        return self._learning_rate

    def _gather_flat_grad(self, gradients):
        """
                Flatten all gradients into a single 1D tensor.

                Args:
                    gradients: List of gradient tensors.

                Returns:
                    A single 1D tensor containing all gradients concatenated together.
        """
        return tf.concat([tf.reshape(g, [-1]) for g in gradients if g is not None], axis=0)

    def _apply_update(self, params, update):
        """
                Apply the computed update to the model parameters.

                Args:
                    params: List of model parameters.
                    update: Flattened update values.
        """
        offset = 0
        for p in params:
            shape = tf.shape(p)
            numel = tf.reduce_prod(shape)
            update_slice = tf.reshape(update[offset:offset + numel], shape)
            offset += numel
            p.assign_add(update_slice)

    def _clone_param(self, params):
        """
                Create a copy of the model parameters.

                Args:
                    params: List of model parameters.

                Returns:
                    A list of cloned parameters.
        """
        return [tf.identity(p) for p in params]

    def _set_param(self, params, new_params):
        """
                Replace the current parameters with new values.

                Args:
                    params: List of current model parameters.
                    new_params: List of new parameter values.
        """
        for p, new_p in zip(params, new_params):
            p.assign(new_p)


    def _filter_none_grads(self, params, grads):
        """
                Remove parameters with None gradients.

                Args:
                    params: List of model parameters.
                    grads: List of gradients.

                Returns:
                    Tuple of filtered parameters and gradients.
        """
        filtered_params = []
        filtered_grads = []
        for p, g in zip(params, grads):
            if g is not None:
                filtered_params.append(p)
                filtered_grads.append(g)
        return filtered_params, filtered_grads

    def _directional_evaluate(self, closure, params, x_init, t, d):
        """
                Evaluate the loss and gradient along a given search direction.

                Args:
                    closure: Callable function to compute the loss.
                    params: List of model parameters.
                    x_init: Initial parameters.
                    t: Step size.
                    d: Search direction.

                Returns:
                    Tuple containing the loss and flattened gradient.
        """

        self._apply_update(params, t * d) # Update parameters temporarily

        with tf.GradientTape() as tape:
            loss = closure() # Compute the loss
        grads = tape.gradient(loss, params)
        params, grads = self._filter_none_grads(params, grads)
        flat_grad = self._gather_flat_grad(grads)

        self._set_param(params, x_init) # Restore parameters to original values
        return float(loss.numpy()), flat_grad


    def _compute_direction(self, gradients):
        """
                Compute the search direction using the L-BFGS two-loop recursion.

                Args:
                    gradients: Flattened gradient tensor.

                Returns:
                    A tensor representing the search direction.
        """
        state = self.state
        if state['n_iter'] == 1:
            # First iteration: use steepest descent direction
            direction = -gradients
            state['old_dirs'] = []
            state['old_stps'] = []
            state['ro'] = []
            state['H_diag'] = 1.0
        else:
            # Update direction using L-BFGS two-loop recursion
            y = gradients - state['prev_flat_grad']
            s = state['direction'] * state['t']
            ys = tf.tensordot(y, s, axes=1)
            # Update history if curvature condition is satisfied
            if ys > self.tolerance_ys:
                if len(state['old_dirs']) == self.history_size:
                    state['old_dirs'].pop(0)
                    state['old_stps'].pop(0)
                    state['ro'].pop(0)
                state['old_dirs'].append(y)
                state['old_stps'].append(s)
                state['ro'].append(1.0 / ys)
                state['H_diag'] = ys / tf.tensordot(y, y, axes=1)

            # Perform two-loop recursion to compute search direction
            q = -gradients
            alphas = []
            for old_dir, old_stp, ro in zip(reversed(state['old_dirs']), reversed(state['old_stps']), reversed(state['ro'])):
                alpha = ro * tf.tensordot(old_stp, q, axes=1)
                q -= alpha * old_dir
                alphas.append(alpha)

            direction = q * state['H_diag']
            for old_dir, old_stp, ro, alpha in zip(state['old_dirs'], state['old_stps'], state['ro'], reversed(alphas)):
                beta = ro * tf.tensordot(old_dir, direction, axes=1)
                direction += old_stp * (alpha - beta)

        state['direction'] = direction
        state['prev_flat_grad'] = gradients
        return direction

    def minimize(self, loss_fn, var_list):
        """
                Minimize the loss function using the L-BFGS optimization algorithm.

                Args:
                    loss_fn: Callable
                        Function to compute the loss.
                    var_list: List
                        List of model parameters to optimize.
        """
        params = var_list
        # Evaluate the initial loss and gradients
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, params)
        ##########################
        params, grads = self._filter_none_grads(params, grads)
        ################
        flat_grad = self._gather_flat_grad(grads)
        self.state['func_evals'] += 1
        self.state['n_iter'] += 1

        if tf.reduce_max(tf.abs(flat_grad)) <= self.tolerance_grad:
            return loss

        # Initial direction
        direction = self._compute_direction(flat_grad)
        if self.state['n_iter'] == 1:
            t = tf.minimum(1.0, 1.0 / tf.reduce_sum(tf.abs(flat_grad))) * self.learning_rate
        else:
            t = self.learning_rate

        prev_loss = float(loss)
        x_init = self._clone_param(params)

        # Main optimization loop
        n_iter = 0
        current_evals = 1
        while n_iter < self.max_iter:
            n_iter += 1

            gtd = tf.tensordot(flat_grad, direction, axes=1)
            # If directional derivative >= -tolerance_change, break
            if gtd > -self.tolerance_change:
                break

            # Line search
            if self.line_search_fn == 'strong_wolfe':
                def obj_func(x_params, step, dir):
                    return self._directional_evaluate(loss_fn, params, x_params, step, dir)

                f_val, g_new, t, ls_func_evals = _strong_wolfe(obj_func, x_init, t, direction, float(loss), flat_grad, float(gtd),
                                                               c1=1e-4, c2=0.9, tolerance_change=self.tolerance_change, max_ls=25)
                self._apply_update(params, t * direction)
                flat_grad = g_new
                loss = f_val
            else:
                # If no line search, just do a fixed step
                self._apply_update(params, t * direction)
                loss = loss_fn()
                grads = tf.gradients(loss, params)
                flat_grad = self._gather_flat_grad(grads)
                ls_func_evals = 1

            current_evals += ls_func_evals
            self.state['func_evals'] += ls_func_evals

            # Check convergence conditions
            if tf.reduce_max(tf.abs(flat_grad)) <= self.tolerance_grad:
                break

            # Parameter change too small?
            if tf.reduce_max(tf.abs(direction * t)) <= self.tolerance_change:
                break

            # Function value change too small?
            if abs(float(loss) - prev_loss) < self.tolerance_change:
                break

            if current_evals >= self.max_eval:
                break

            prev_loss = float(loss)
            self.state['t'] = t
            direction = self._compute_direction(flat_grad)
            x_init = self._clone_param(params)

        self.state['prev_loss'] = prev_loss
        return loss
