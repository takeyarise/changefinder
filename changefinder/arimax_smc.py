import numpy as np


class ARMAXusingSMC(object):
    r"""ARMAX モデル．パラメータは SMC 法により求める．

    $$
    y_t
    = \mu
    + \sum_{i=1}^p \alpha_i (y_{t-i} - \mu)
    + \sum_{i=1}^q \beta_i \epsilon_{t-i}
    + \sum_{i=1}^M \gamma_i X_{i,t}
    + \epsilon_t
    \\
    \epsilon_t\sim N(0, \sigma^2)
    $$

    $\mu_t, \sigma^2_t, \alpha_{1:p,t}, \beta_{1:q, t}, \gamma_{1:M,t}$ は
    それぞれ各 1 時点前のパラメータ分布の期待値と分散一定の正規分布より生成する．
    分散の大きさはそれぞれのパラメータに応じて設定する．
    """

    def __init__(self, p, q, M, n_particle):
        assert isinstance(p, int) and p > 0
        assert isinstance(q, int) and q > 0
        assert isinstance(M, int) and M >= 0
        assert isinstance(n_particle, int) and n_particle > 0

        self.__p = p
        self.__q = q
        self.__M = M
        self.__n_particle = n_particle

        self.__mu = np.random.normal(0, 1, size=(n_particle,))
        self.__alpha = np.random.normal(0, 1, size=(n_particle, p))
        self.__beta = np.random.normal(0, 1, size=(n_particle, q))
        self.__gamma = np.random.normal(0, 1, size=(n_particle, M))
        self.__sigma = np.random.gamma(1.0, 1.0, size=(n_particle,))

        self.__obs = list()
        self.__eps = list()

    def __push_list(self, obs, series, max_size):
        series.append(obs)
        if len(series) > max_size:
            series.pop(0)

    def __push_obs(self, obs):
        self.__push_list(obs, self.__obs, self.__p)

    def __push_eps(self, eps):
        self.__push_list(eps, self.__eps, self.__q)

    def predict(self):
        """predict \hat{x}_t from parametes.
        """
        pass

    def likelihood(self, obs, pred, s2):
        return np.exp(- (obs - pred)**2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)

    def update(self, obs):
        """update.

        Parameters
        ----------
        obs: observation
        """
        pass


class ChangeFinderARMAXusingSMC(object):
    def __init__(self):
        pass

    def __add_one(self, element, time_series, size):
        time_series.append(element)
        if len(time_series) == size + 1:
            time_series.pop(0)
        return

    def update(self, x):
        # first step
        # smoothing
        # second step
        pass