import random
import numpy as np


class NormalModel(object):
    def __init__(self):
        pass

    def predict(self, mu, sigma, size):
        return np.random.normal(mu, sigma, size=size)

    def likelihood(self, obs, pred, sigma):
        s2 = sigma ** 2
        return np.exp(- (obs - pred) ** 2 / (2 * s2)) / np.sqrt(2 * np.pi * s2)


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

        self.__mu = self.__initial_param(0.0, 1.0)
        self.__alpha = [
            self.__initial_param(0.0, 1.0)
            for _ in range(p)
        ]
        self.__beta = [
            self.__initial_param(0.0, 1.0)
            for _ in range(q)
        ]
        self.__gamma = [
            self.__initial_param(0.0, 1.0)
            for _ in range(M)
        ]
        self.__sigma = self.__initial_param(0.0, 1.0)
        self.__sigma['values'] = np.random.gamma(1.0, 1.0, size=(n_particle,))

        self.__obs = list()  # 先頭から順に 1, 2,..., t
        self.__eps = list()  # となりインデックスと逆順になることに注意

        self.__model = NormalModel()

    def __initial_param(self, mu, sigma):
        return {
            'values': np.random.normal(mu, sigma, size=(self.__n_particle,)),
            'weights': np.ones(self.__n_particle),
            'sigma': sigma,
        }

    def __push_list(self, obs, series, max_size):
        series.append(obs)
        if len(series) > max_size:
            series.pop(0)

    def __push_obs(self, obs):
        self.__push_list(obs, self.__obs, self.__p)

    def __push_eps(self, eps):
        self.__push_list(eps, self.__eps, self.__q)

    def __term(self, param, is_particle):
        if is_particle:
            return param['values']
        else:
            return np.average(param['values'], weights=param['weights'])

    def __predict(self, eps, p_name, p_idx=None, exog=None):
        """predict \hat{x}_t from parametes.

        Returns
        -------
        \hat{x}_t
        epsilon_t
        """
        pred = 0
        mu = self.__term(self.__mu, p_name == 'mu')
        pred += mu
        for i in range(self.__p):
            pred += self.__term(self.__alpha[i], (p_name == 'alpha') and (i == p_idx)) * (self.__obs[i] - mu)
        for i in range(self.__q):
            pred += self.__term(self.__beta[i], (p_name == 'beta') and (i == p_idx)) * self.__eps[i]
        for i in range(self.__M):
            pred += self.__term(self.__gamma[i], (p_name == 'gamma') and (i == p_idx)) * exog[i]
        pred += eps
        return pred

    def predict(self, exog=None, eps=None):
        """t + 1 時刻目の予測"""
        if eps is None:
            eps = self.__model.predict(
                0,
                self.__term(self.__sigma, False),
                size=1,
            )
        return self.__predict(eps, '', None, exog)

    def __predict_w_params(self, exog):
        mu = self.__mu['values'].copy()
        pred = mu
        for i in range(self.__p):
            pred += self.__alpha[i]['values'] * (self.__obs[i] - mu)
        for i in range(self.__q):
            pred += self.__beta[i]['values'] * self.__eps[i]
        for i in range(self.__M):
            pred += self.__gamma[i]['values'] * exog[i]
        pred += self.__model.predict(0, self.__sigma['values'], None)
        return pred

    def __resampling(self, particles, weights):
        normed = np.cumsum(weights) / np.sum(weights)
        return np.array(
            random.choices(
                particles,
                cum_weights=normed,
                k=self.__n_particle,
            )
        )

    def __param_update(self, obs, eps, exog=None):
        # 重みの計算
        mu_w = self.__model.likelihood(
            obs,
            self.__predict(eps, 'mu', exog=exog),
            self.__mu['sigma']
        )
        alpha_w = [
            self.__model.likelihood(
                obs,
                self.__predict(eps, 'alpha', i, exog=exog),
                self.__alpha[i]['sigma']
            ) for i in range(self.__p)
        ]
        beta_w = [
            self.__model.likelihood(
                obs,
                self.__predict(eps, 'beta', i, exog=exog),
                self.__beta[i]['sigma']
            ) for i in range(self.__q)
        ]
        gamma_w = [
            self.__model.likelihood(
                obs,
                self.__predict(eps, 'gamma', i, exog=exog),
                self.__gamma[i]['sigma']
            ) for i in range(self.__M)
        ]
        sigma_w = self.__model.likelihood(
            obs,
            self.__predict(self.__model.predict(0, self.__sigma['values']), 'sigma', exog=exog),
            self.__sigma['sigma']
        )
        # 重みの更新
        self.__mu['weights'] = np.exp(np.log(self.__mu['weights']) + np.log(mu_w))
        for i in range(self.__p):
            self.__alpha[i]['weights'] = np.exp(np.log(self.__alpha[i]['weights']) + np.log(alpha_w[i]))
        for i in range(self.__q):
            self.__beta[i]['weights'] = np.exp(np.log(self.__beta[i]['weights']) + np.log(beta_w[i]))
        for i in range(self.__M):
            self.__gamma[i]['weights'] = np.exp(np.log(self.__gamma[i]['weights']) + np.log(gamma_w[i]))
        self.__sigma['weights'] = np.exp(np.log(self.__sigma['weights']) + np.log(sigma_w))
        # リサンプリング
        # （論文によると w が一定以上でリサンプリングした方がいいらしい）
        self.__mu['values'] = self.__resampling(self.__mu['values'], self.__mu['weights'])
        self.__sigma['values'] = self.__resampling(self.__sigma['values'], self.__sigma['weights'])
        for i in range(self.__p):
            self.__alpha[i]['values'] = self.__resampling(self.__alpha[i]['values'], self.__alpha[i]['weights'])
        for i in range(self.__q):
            self.__beta[i]['values'] = self.__resampling(self.__beta[i]['values'], self.__beta[i]['weights'])
        for i in range(self.__M):
            self.__gamma[i]['values'] = self.__resampling(self.__gamma[i]['values'], self.__gamma[i]['weights'])
        # モデルパラメータの最高神
        self.__mu['values'] += self.__model.predict(0, self.__mu['sigma'], (self.__n_particle,))
        self.__sigma['values'] += self.__model.predict(0, self.__sigma['sigma'], (self.__n_particle,))
        for i in range(self.__p):
            self.__alpha[i]['values'] += self.__model.predict(0, self.__alpha[i]['sigma'], (self.__n_particle,))
        for i in range(self.__q):
            self.__beta[i]['values'] += self.__model.predict(0, self.__beta[i]['sigma'], (self.__n_particle,))
        for i in range(self.__M):
            self.__gamma[i]['values'] += self.__model.predict(0, self.__gamma[i]['sigma'], (self.__n_particle,))

    def update(self, obs, exog=None):
        """update.

        Parameters
        ----------
        obs: observation
        exos: exogenous variables

        Returns
        -------
        $-\log p_{t-1}(x_t)$
        """
        eps = self.__model.predict(
            0,
            self.__term(self.__sigma, False),
            size=1,
        )
        ys_hat = self.__predict_w_params(exog)

        self.__param_update(obs, eps, exog=exog)
        self.__push_obs(obs)
        self.__push_eps(eps)

        weight = self.__model.likelihood(obs, ys_hat, self.__sigma['values'])
        return - np.log(np.sum(weight))


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