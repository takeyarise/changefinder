from .particle import NormalTransition, Particle
import copy
import numpy as np


class Transition(object):
    def __init__(self, sigma):
        self.__sigma = sigma

    def predict(self, current):
        ret = current + np.random.normal(0, self.__sigma, size=current.shape)
        sigma = ret[:, 1]  # NOTE: とりまクリッピング
        sigma[sigma < 0] = 0
        return ret


class SMCARMAX(object):
    def __init__(self, p, q, m, num_particle):
        assert isinstance(p, int) and p > 0
        assert isinstance(q, int) and q > 0
        assert isinstance(m, int) and m >= 0
        assert isinstance(num_particle, int) and num_particle > 0

        self.p = p
        self.q = q
        self.m = m
        self.num_particle = num_particle

        init_params = np.random.normal(0, 1, size=(num_particle, 2 + p + q + m))
        init_params[:, 1] = np.random.gamma(1.0, 1.0, size=(num_particle,))
        self.params = Particle(
            num_particle,
            2 + p + q + m,  # mu, sigma, alpha_{1:p}, beta_{1:q}, gamma_{1:m} の順とする
            Transition(sigma=1),  # とりますべてのパラメータの標準偏差を 1 とする．
            self,
            initial=init_params,
        )

        # とりま 0 埋め
        self.obs = list()
        self.eps = list(np.random.normal(0, 1, size=p))
        self.exog = None

    def predict(self, eps=None, exog=None):
        if len(self.obs) < self.p:
            return None
        params = self.params.particles.copy()
        i = 0
        pred = params[:, i].copy()
        i = 1
        if eps is None:
            eps = copy.copy(self.eps[-1])
        pred = pred + eps
        i = 2
        pred = pred + np.sum(params[:, i:i+self.p].T * (np.array(self.obs) - params[:, 0]), axis=0)
        i = i + self.p
        pred = pred + np.sum(params[:, i:i+self.q].T * np.array(self.eps), axis=0)
        i = i + self.q
        if exog is not None:
            pred = pred + np.sum(params[:, i:i+self.m].T * np.array(exog), axis=0)
        return pred

    def likelihood(self, _, obs):
        """対数尤度をとってみる"""
        preds = self.predict(self.exog)
        s2 = self.params.particles[:, 1].copy()
        s2 = s2 ** 2
        #ret = np.exp(- ((obs - preds)**2) / (2 * s2)) / np.sqrt(2 * np.pi * s2)
        term1 = - (((obs - preds)**2) / (2 * s2))
        term2 = - np.log(2 * np.pi * s2) / 2
        return term1 + term2

    def __call__(self, preds, obs):
        return self.likelihood(preds, obs)

    def push_list(self, obs, series, max_size):
        series.append(obs)
        if len(series) > max_size:
            series.pop(0)

    def push_obs(self, obs):
        self.push_list(obs, self.obs, self.p)

    def push_eps(self, eps):
        self.push_list(eps, self.eps, self.q)

    def update(self, obs, exog=None):
        """update.

        Parameters
        ----------
        obs: observation
        exos: exogenous variables, list

        Returns
        -------
        $E[p_{t-1}(x_t)]$
        """
        if len(self.obs) < self.p:
            self.push_obs(obs)
            return None
        self.exog = exog
        eps = np.random.normal(0, self.params.particles[:, 1])
        self.push_eps(eps)
        next_pred = self.predict(eps, exog)
        next_params, weights = self.params.update(obs)
        self.params.resampling(next_params, weights)
        self.push_obs(obs)
        return np.average(next_pred, weights=weights, axis=0)
