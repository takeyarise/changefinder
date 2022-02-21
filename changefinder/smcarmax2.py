from .particle import Particle
import copy
from functools import partial
import numpy as np


class Transition(object):
    def __init__(self, sigma):
        self.__sigma = sigma

    def predict(self, current):
        ret = current + np.random.normal(0, self.__sigma, size=current.shape)
        return ret

    def __call__(self, current):
        return self.predict(current)


class SigmaTransition(object):
    def __init__(self, sigma):
        self.__sigma = sigma

    def predict(self, current):
        ret = current + np.random.normal(0, self.__sigma, size=current.shape)
        return ret.clip(min=1e-6)

    def __call__(self, current):
        return self.predict(current)


class SMCARMAX2(object):
    def __init__(self, p, q, m, num_particle, **kwargs):
        assert isinstance(p, int) and p >= 0
        assert isinstance(q, int) and q >= 0
        assert isinstance(m, int) and m >= 0
        assert isinstance(num_particle, int) and num_particle > 0

        self.p = p
        self.q = q
        self.m = m
        self.num_particle = num_particle

        self.mu = self.__set_initial_param('mu', kwargs.pop('mu_sigma', 1))
        self.sigma = self.__set_initial_param('sigma', kwargs.pop('sigma_sigma', 1), True)
        self.alpha = [self.__set_initial_param(f'alpha_{i}', kwargs.pop(f'alpha_{i}_sigma', 1)) for i in range(p)]
        self.beta = [self.__set_initial_param(f'beta_{i}', kwargs.pop(f'beta_{i}_sigma', 1)) for i in range(q)]
        self.gamma = [self.__set_initial_param(f'gamma_{i}', kwargs.pop(f'gamma_{i}_sigma', 1)) for i in range(m)]
        self.obs = list()
        self.eps = list(np.random.normal(0, 1, size=(p + 1)))
        self.exog = None

    def __set_initial_param(self, name, sigma=1, is_sigma=False):
        transition_func = Transition(sigma=sigma)
        init = None

        if is_sigma:
            transition_func = SigmaTransition(sigma=sigma)
            init = np.random.gamma(1.0, 1.0, size=(self.num_particle, 1))

        return Particle(
            self.num_particle,
            1,
            transition_func,
            partial(self.likelihood, update_param_name=name),
            initial=init,
        )

    def push_list(self, obs, series, max_size):
        series.append(obs)
        if len(series) > max_size:
            series.pop(0)

    def push_obs(self, obs):
        self.push_list(obs, self.obs, self.p)

    def push_eps(self, eps):
        self.push_list(eps, self.eps, self.q)

    def likelihood(self, param, obs, update_param_name):
        """対数尤度を返す"""
        if param is not None:
            param = param.copy()
        y = self.predict(update_param_name=update_param_name, update_param=param)
        if self.check_param_name(update_param_name, 'sigma'):
            s2 = param ** 2
        else:
            s2 = self.sigma.expected() ** 2
        term1 = - (obs - y)**2 / (2 * s2)
        term2 = - np.log(2 * np.pi * s2) / 2
        return term1 + term2

    def check_param_name(self, src_name, dst_name):
        return (src_name is not None) and (src_name == dst_name)

    def predict(self, exog=None, eps=None, **kwargs):
        update_param_name = kwargs.pop('update_param_name', None)
        update_param = kwargs.pop('update_param', None)
        if len(self.obs) < self.p:
            return None
        if eps is None:
            eps = self.eps[-1]
        if exog is None:
            exog = self.exog

        if self.check_param_name(update_param_name, 'mu'):
            mu = update_param
        else:
            mu = self.mu.expected()
        if self.check_param_name(update_param_name, 'sigma'):
            sigma = np.random.normal(0, update_param)
        else:
            sigma = eps
        alpha = list()
        for i in range(self.p):
            if self.check_param_name(update_param_name, 'alpha_{}'.format(i)):
                t = update_param * (self.obs[i] - mu)
            else:
                t = self.alpha[i].expected() * (self.obs[i] - mu)
            alpha.append(t)
        beta = list()
        for i in range(self.q):
            if self.check_param_name(update_param_name, 'beta_{}'.format(i)):
                t = update_param * (self.eps[i])
            else:
                t = self.beta[i].expected() * (self.eps[i])
            beta.append(t)
        gamma = list()
        for i in range(self.m):
            if self.check_param_name(update_param_name, 'gamma_{}'.format(i)):
                t = update_param * (exog[i])
            else:
                t = self.gamma[i].expected() * (exog[i])
            gamma.append(t)

        pred = mu + sigma + sum(alpha) + sum(beta) + sum(gamma)
        return pred

    def update(self, obs, exog=None):
        if len(self.obs) < self.p:
            self.push_obs(obs)
            return np.nan, np.nan
        self.exog = exog
        self.push_eps(np.random.normal(0, self.sigma.expected()))
        # パラメータの更新
        yt = self.predict(exog)
        pt = self.likelihood(None, obs, None)
        mu = self.mu.calc_next_particle(obs)
        sigma = self.sigma.calc_next_particle(obs)
        alpha = list()
        for i in range(self.p):
            alpha.append(self.alpha[i].calc_next_particle(obs))
        beta = list()
        for i in range(self.q):
            beta.append(self.beta[i].calc_next_particle(obs))
        gamma = list()
        for i in range(self.m):
            gamma.append(self.gamma[i].calc_next_particle(obs))
        # リサンプリング
        self.mu.resampling(*mu)
        self.sigma.resampling(*sigma)
        for i in range(self.p):
            self.alpha[i].resampling(*(alpha[i]))
        for i in range(self.q):
            self.beta[i].resampling(*(beta[i]))
        for i in range(self.m):
            self.gamma[i].resampling(*(gamma[i]))
        # その他処理
        self.push_obs(obs)
        return yt, pt