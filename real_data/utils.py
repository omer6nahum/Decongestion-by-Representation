import numpy as np
import cvxpy as cp
from collections import Counter
from scipy.special import softmax
from sklearn.decomposition import NMF
from cvxopt import solvers
import tensorflow as tf


EPSILON = np.finfo(tf.float32.as_numpy_dtype).tiny
solvers.options['show_progress'] = False


NO_ITEM = -1


def param_str(param_name, param, default_val):
    return f'__{param_name}_{param:.2f}'.replace('.', '-') if param != default_val else ''


def temp_dist(scores, T):
    return softmax(scores / T)


def factorize(B, n_components=3):
    nmf = NMF(n_components=n_components, max_iter=1000)
    U = nmf.fit_transform(B)
    T = nmf.components_
    return U, T


# =========
# choice and allocation
# =========
def choice(V, p):
    utility = V - p
    y = np.argmax(utility, axis=1)
    # if all utilities are non-positive, the choice would be NO_ITEM
    users_all_negative = np.all(utility <= 0, axis=1)
    y[users_all_negative] = NO_ITEM
    return y


def shift_choice(y):
    # shift all chosen items by 1, so item 0 is actually NO_ITEM
    return np.array(y) + 1


def uniform_alloc(y, m):
    # y is a choice profile (y1, ... , yn)
    n = len(y)
    a = np.zeros((n, m))
    for item in np.unique(y):
        if item == NO_ITEM:
            continue
        potential_users = np.where(y == item)[0]
        # update results
        a[potential_users, item] = 1 / len(potential_users)

    assert np.all(np.sum(a, axis=0) <= 1 + 1e-10)
    assert np.all(np.sum(a, axis=1) <= 1 + 1e-10)
    return a


# =========
# metrics
# =========
def welfare(V, a):
    return np.sum(V * a)


def n_chosen_perc(y, m):
    chosen = set(y)
    try:
        chosen.remove(NO_ITEM)
    except KeyError:
        pass
    return 100*len(chosen)/m


def n_no_item_chosen(y):
    return 100 * np.mean(y == NO_ITEM)


def congestion(y):
    # over all items, sum the total number of over-congested choices
    total_sum = 0
    choices_per_item = Counter(y)
    for item, count in choices_per_item.items():
        if item == NO_ITEM:
            total_sum += count  # also penalize for no-choice
            continue
        total_sum += max((0, count - 1))

    return total_sum


def calc_all_metrics(X, U, T, p, mask):
    """
    Compute metrics for a mask for markets (X, U, p) based on true preferences
    :param X: items features (m, d)
    :param U: users features (n_markets, n, d)
    :param T: true transformation between U and B
    :param p: prices (n_markets, m)
    :param mask: binary mask for items (same mask for all items) (m, d)
    :return:
    """
    m, d = X.shape
    n_markets = U.shape[0]
    welfares = []
    congestions = []
    null_items = []
    unique_items = []

    for j in range(n_markets):
        Z = X * mask
        B = U[j] @ T
        V = B @ X.T
        V_tilde = B @ (Z.T)
        y = choice(V_tilde, p[j])
        a = uniform_alloc(y, m)
        w = welfare(V, a)
        welfares.append(w)
        congestions.append(congestion(y))
        null_items.append(n_no_item_chosen(y))
        unique_items.append(n_chosen_perc(y, m))
    
    return {'welfare': np.mean(welfares),
            'congestion': np.mean(congestions),
            'null_items': np.mean(null_items),
            'unique_items': np.mean(unique_items)
           }


# =========
# masking
# =========
def uniform_mask(k, d, m):
    assert k <= d
    mask = np.zeros((m, d))
    permutation = np.random.permutation(range(d))
    mask[:, permutation[:k]] = 1
    return mask


def sample_mask(scores, k, m, T=1):
    # scores is a np.array of the features' distribution (un-normalized)
    # T (> 0) is for temperature - the lower temperature is, the distribution gets more extreme.
    # a temperature of T=infinity is a uniform distribution.
    # k is the desired mask size
    # m is the number of items
    d = len(scores)
    
    # calc a distribution vector (softmax on (temperatured) scores)
    dist = softmax(scores / T)
    
    chosen_features = []
    for i in range(k):
        dist /= np.sum(dist)
        assert np.abs(np.sum(dist) - 1) <= 1e-5  # sums up to 1
        # sample from current distribution
        feature_i = np.argmax(np.random.multinomial(n=1, pvals=dist))
        chosen_features.append(feature_i)
        # adjust distribution given the chosen features so far
        dist[feature_i] = 0
    
    mask = np.zeros((m, d))
    mask[:, chosen_features] = 1
    return mask


# =========
# prices
# =========
def set_prices_primal(V, return_val=False):
    # find CE prices for V
    n = V.shape[0]
    assert(n == V.shape[1])
    assert isinstance(return_val, bool), f'return val ({return_val}) is not bool'
    a = cp.Variable((n, n))
    welfare = cp.sum(cp.multiply(V, a).flatten())
    unit_demand = cp.sum(a, 0) <= 1
    unit_supply = cp.sum(a, 1) <= 1
    ge0 = a >= 0
    le1 = a <= 1
    constraints = [unit_demand, unit_supply, ge0, le1]
    prob = cp.Problem(cp.Maximize(welfare), constraints)
    prob.solve()
    opt_val = prob.value
    p_primal = constraints[0].dual_value
    
    if return_val:
        return p_primal, opt_val
    else:
        return p_primal
    
    
def prices_min(V, opt_val=None):
    # buyer optimal prices
    n = V.shape[0]
    assert(n == V.shape[1])
    if opt_val is None:
        _, opt_val = set_prices_primal(V, return_val=True)
    s = cp.Variable(n)
    b = cp.Variable(n)
    dual_obj = cp.sum(s)
    dual_ge0_s = s >= 0
    dual_ge0_b = b >= 0
    dual_constraints = [dual_ge0_s, dual_ge0_b]
    for i in range(n):
        for j in range(n):
            dual_constraints.append(s[i] + b[j] >= V[j, i])
    dual_constraints.append(cp.sum(s) + cp.sum(b) == opt_val)
    dual_prob = cp.Problem(cp.Minimize(dual_obj), dual_constraints)
    dual_prob.solve()
    p_min = s.value
    return p_min


def prices_max(V, opt_val=None):
    # seller optimal prices
    n = V.shape[0]
    assert(n == V.shape[1])
    if opt_val is None:
        _, opt_val = set_prices_primal(V, return_val=True)
    s = cp.Variable(n)
    b = cp.Variable(n)
    dual_obj = cp.sum(b)
    dual_ge0_s = s >= 0
    dual_ge0_b = b >= 0
    dual_constraints = [dual_ge0_s, dual_ge0_b]
    for i in range(n):
        for j in range(n):
            dual_constraints.append(s[i] + b[j] >= V[j, i])
    dual_constraints.append(cp.sum(s) + cp.sum(b) == opt_val)
    dual_prob = cp.Problem(cp.Minimize(dual_obj), dual_constraints)
    dual_prob.solve()
    p_max = s.value
    return p_max
    
    
def set_prices(V, gamma=0.5, V_epsilon=0.0, p_epsilon=0.0, rho=0.0):
    # default params values produce CE prices

    # add noise to V
    V_noise = np.random.normal(0, V_epsilon, size=V.shape)
    V = np.array(V) + V_noise
    V[V < 0] = 0
    V[V > 1] = 1
    
    # two linear lines.
    # gamma=0.0 is buyer_opt, 0.5 is primal solution, 1.0 is seller opt
    p_primal, sol_val = set_prices_primal(V, return_val=True)
    if gamma < 0.5:
        p_min = prices_min(V, sol_val)
        new_gamma = 2 * gamma
        p = ((1 - new_gamma) * p_min) + (new_gamma * p_primal)
    elif gamma > 0.5:
        p_max = prices_max(V, sol_val)
        new_gamma = (gamma - 0.5) * 2
        p = ((1 - new_gamma) * p_primal) + (new_gamma * p_max)
    else:
        p = p_primal
    
    # convex comb between CE prices to heuristic prices.
    # rho=0.0 is CE prices, rho=1.0 is mean-value prices.
    if rho > 0.0:
        p_hue = V.mean(axis=0)
        p = (rho * p_hue) + ((1 - rho) * p)
    
    # add noise to p
    p_noise = np.random.normal(0, p_epsilon, size=p.shape)
    p = p + p_noise
    p[p < 0] = 0
    p[p > 1] = 1
    
    return p


# =========
# Gumbel
# Gumbel functions and layer are based on / taken from
# https://github.com/ermongroup/subsets/blob/master/subsets/sample_subsets.py ,
# the code directory of "Reparameterizable Subset Sampling via Continuous Relaxations"
# (https://arxiv.org/pdf/1901.10517.pdf)
# =========
def gumbel_keys(w):
    uniform = tf.random.uniform(
        tf.shape(w),
        minval=EPSILON,
        maxval=1.0)
    z = -tf.math.log(-tf.math.log(uniform))
    return w + z


def continuous_topk(w, k, t):
    khot_list = []
    onehot_approx = tf.zeros_like(w, dtype=tf.float32)
    for i in range(k):
        khot_mask = tf.maximum(1.0 - onehot_approx, EPSILON)
        w += tf.math.log(khot_mask)
        onehot_approx = tf.nn.softmax(w / t, axis=-1)
        khot_list.append(onehot_approx)
    return tf.reduce_sum(khot_list, 0)


def sample_subset(w, k, t=0.1, hard=False):
    """
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
        hard (bool): return a top-k vector if hard=True, relaxed-top-k otherwise
    """
    d = len(w)
    flip = (k > (d / 2)) and (k != d)
    w = gumbel_keys(w)
    if flip:
        sample = continuous_topk(w, d - k, t).numpy()
        sample = 1 - sample
    else:
        sample = continuous_topk(w, k, t).numpy()

    if hard:
        indices = np.argsort(sample)
        sample[indices[-k:]] = 1
        sample[indices[:-k]] = 0
    return sample


def mean_sample_subset(w, k, t=0.1, n_samples=100, hard=True):
    # (!) w is a probability vector
    samples = []
    for i in range(n_samples):
        sample = np.array(sample_subset(w=np.log(w), k=k, t=t, hard=hard))
        samples.append(sample)
    res = np.mean(samples, axis=0)
    return res / res.sum()


class GumbelK(tf.keras.layers.Layer):
    # define trainable distribution parameters w
    def __init__(self, dim, k, N, t=0.01, weights_t=1, **kwargs):
        """
        dim - weights dimension (number of parameters for distribution)
        k - size of subset (k from top-k)
        N - number of masks to sample from the distribution
        t - temperature for distribution
        """
        super().__init__(**kwargs)
        self.k = k
        self.t = t
        self.weights_t = weights_t
        self.N = N
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(dim,)) / dim,
            trainable=True,
        )
        self.masks = []

        self.flip = False
        if (self.k > (dim / 2)) and (self.k != dim):
            self.flip = True
            self.k = dim - self.k

    def call(self, inputs):
        res = []
        log_proba = tf.math.log(tf.nn.softmax(self.w / self.weights_t))
        for i in range(self.N):  # N samples of masks
            r = gumbel_keys(log_proba)
            res.append(continuous_topk(r, self.k, self.t))
        res = tf.stack(res)  # (N, dim)
        if self.flip:
            res = 1 - res
        self.masks.append(res)
        return res


# =========
# loss
# =========
class LossLayer(tf.keras.layers.Layer):
    def __init__(self, lam=1, name=None):
        super().__init__(name=name)
        self.lam = lam  # balancing coefficient in the proxy loss
        self.losses_per_masks = []

    def call(self, inp_p, Y):
        """
            inp_p - prices
                (batch, m+1)
            Y - represents choices of (n) users on (m+1) items in multiple (batch) markets with multiple (N) masks samples
                (batch, N, n, m+1)
        """
        # first_summand ("selection")
        sum_on_j = tf.einsum('bkij,bj->bki', Y, inp_p)  # (batch, N, n)
        sum_on_ij = tf.reduce_sum(sum_on_j, axis=2)  # (batch, N)
        first_summand = tf.reduce_mean(sum_on_ij)  # scalar

        # second_summand ("congestion")
        # [:,:,:,1:] (do not penalize for no-choice in this term)
        demand = tf.math.reduce_sum(Y[:, :, :, 1:], axis=2)  # (batch, N, m)   # sum on i
        congestion = tf.math.minimum(0.0, 1 - demand)  # (batch, N, m)
        sum_on_j = tf.reduce_sum(congestion, axis=2)  # (batch, N)
        second_summand = tf.reduce_mean(sum_on_j)  # scalar

        # third_summand ("null item choices")
        null_item_choices = -tf.math.reduce_sum(Y[:, :, :, 0], axis=2)  # (batch, N)   # sum on i
        third_summand = tf.reduce_mean(null_item_choices)

        proxy_welfare = ((1 - self.lam) * first_summand) + (self.lam * second_summand) \
                        + (self.lam * third_summand)
        loss = -proxy_welfare

        self.add_loss(loss)
        self.add_metric(first_summand, 'proxy_1st')
        self.add_metric(second_summand, 'proxy_2nd')
        self.add_metric(third_summand, 'proxy_3rd')

        proxy_welfare_per_mask = ((1 - self.lam) * sum_on_ij) + (self.lam * sum_on_j) \
                                 + (self.lam * null_item_choices)
        self.losses_per_masks.append(-proxy_welfare_per_mask)

        # return loss
        return Y


def compile_loss(dummy_target, y_pred):
    # a loss function of 0. every other loss is added via add_loss() interface
    return 0


# =========
# callbacks
# =========
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, U, p, X, real_T, k):
        self.model = model
        self.gumbel_index = [i for i, l in enumerate(self.model.layers) if l.name == 'gumbel'][0]
        self.X = X
        self.real_T = real_T
        self.U = U
        self.p = p
        self.k = k
        self.n_markets = U.shape[0]

    def on_train_begin(self, logs=None):
        self.metrics = ['welfare', 'welfare_q025', 'welfare_q975',
                        'congestion', 'null_items', 'unique_items']
        self.results = {metric: [] for metric in self.metrics}

    def on_train_end(self, logs=None):
        self.results = {k: np.array(v) for k, v in self.results.items()}

    def on_epoch_end(self, epoch, logs):
        dist_params = np.array(self.model.layers[self.gumbel_index].weights).squeeze()
        log_proba = np.log(softmax(dist_params))
        epoch_results = {metric: [] for metric in self.metrics if metric not in ['welfare_q025', 'welfare_q975']}

        for i in range(50):
            # sample a mask from dist_params
            mask = sample_subset(w=log_proba, k=self.k, t=0.1, hard=True)
            values = calc_all_metrics(self.X, self.U, self.real_T, self.p, mask)
            for metric_name, val in values.items():
                epoch_results[metric_name].append(val)

        for metric_name, val_list in epoch_results.items():
            self.results[metric_name].append(np.mean(val_list))
        quantiles = np.quantile(epoch_results['welfare'], q=[0.025, 0.975])
        self.results['welfare_q025'].append(quantiles[0])
        self.results['welfare_q975'].append(quantiles[1])


class MasksCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        self.model = model
        self.gumbel_index = [i for i, l in enumerate(self.model.layers) if l.name == 'gumbel'][0]

    def on_train_begin(self, logs):
        self.masks = []

    def on_epoch_end(self, epoch, logs):
        self.masks.append(np.array(self.model.layers[self.gumbel_index].weights))


class AccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, prediction_model, X, real_T, U, p, k, gumbel_temp):
        self.prediction_model = prediction_model
        self.X = X
        self.real_T = real_T
        self.U = U
        self.p = p
        self.m = p.shape[1]
        self.U_cp, self.p_ni_cp = self.adjust_init_cp(U, p)
        self.k = k
        self.n = self.U.shape[1]
        self.gumbel_temp = gumbel_temp
        self.model = model
        self.gumbel_index = [i for i, l in enumerate(self.model.layers) if l.name == 'gumbel'][0]

    def on_train_begin(self, logs):
        self.accuracies = []

    def on_epoch_end(self, epoch, logs):
        w = np.array(self.model.layers[self.gumbel_index].weights).squeeze()
        mu, y = self.create_data(w)
        mu_cp, y_cp = self.adjust_cp(mu, y)
        pred = self.prediction_model.predict([self.p_ni_cp, self.U_cp, mu_cp], batch_size=self.n, verbose=0)
        acc = np.mean(np.argmax(pred, axis=1) == y_cp)
        self.accuracies.append(acc)

    @staticmethod
    def adjust_init_cp(U, p):
        # add null item to p
        p_ni = np.zeros((p.shape[0], p.shape[1] + 1))
        p_ni[:, 1:] = p

        # flatten the market dimension (prediction is independent of other users)
        n = U.shape[1]
        U_cp = U.reshape((-1, U.shape[-1]))

        # repeat mask and prices per user (instead of per market) and flatten the market dimension
        p_ni_cp = np.repeat(p_ni, n, axis=0).reshape((-1, p_ni.shape[-1]))

        return U_cp, p_ni_cp

    @staticmethod
    def adjust_cp(mu, y):
        n = y.shape[1]
        # flatten the market dimension (prediction is independent of other users)
        y_cp = y.flatten()
        # repeat mask and prices per user (instead of per market) and flatten the market dimension
        mu_cp = np.repeat(mu, n, axis=0).reshape((-1, mu.shape[-1]))
        return mu_cp, y_cp

    def create_data(self, w):
        log_proba = np.log(softmax(w))
        all_mu, all_y = [], []

        for i in range(self.U.shape[0]):
            # U[i] is a user matrix (n, d') of the i'th market
            # p[i] is a price vector (m) of the i'th market (for a fixed set of items)
            # calculate V
            B = self.U[i] @ self.real_T
            V = B @ (self.X).T
            # draw a mask mu of size k
            mu = sample_subset(w=log_proba, k=self.k, t=0.1, hard=True)
            # calculate choices of users based on partial information
            Z = self.X * mu
            V_tilde = B @ Z.T
            y = choice(V_tilde, self.p[i])
            # change choices so NO_ITEM is item 0
            y = shift_choice(y)
            all_mu.append(mu)
            all_y.append(y)

        all_mu = np.vstack(all_mu)
        all_y = np.vstack(all_y)

        return all_mu, all_y

