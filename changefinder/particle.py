"""パーティクルフィルタのコードを書く"""
import random
import numpy as np


class NormalTransition(object):
    def __init__(self, sigma):
        self.__sigma = sigma

    def predict(self, current):
        return current + np.random.normal(0, self.__sigma, size=current.shape)


class ObservModel(object):
    def __init__(self):
        pass

    def likelihood(self, preds, obs):
        pass


class Particle(object):
    def __init__(self, num_particle, n_dims, transition_func, likelihood_func, initial=None):
        """

        Parameters
        ----------
        num_particle: int, the number of particles.
        n_dim: int, dimentions.
        transition_func: callable[p_t, (p_{t-1})]
        likelihood_func: callable[weight, (predictions, observation)]
        """
        self.num_particle = num_particle
        self.transition_func = transition_func
        self.likelihood_func = likelihood_func

        if initial is None:
            self.particles = np.zeros((num_particle, n_dims))
        else:
            self.particles = initial

        self.average = np.average(self.particles)
        self.__next = {'param': None, 'weight': None}

    def expected(self):
        return self.average

    def resampling(self, predictions, weights):
        norm_weights = np.cumsum(weights) / np.sum(weights)
        self.particles = np.array(
            random.choices(
                predictions,
                cum_weights=norm_weights,
                k=self.num_particle,
            )
        )

    def update(self, obs):
        preds, weights = self.calc_next_particle(obs)
        self.resampling(preds, weights)
        self.average = np.average(preds, weights=weights, axis=0)
        return preds, weights

    def calc_next_particle(self, obs):
        preds = self.transition_func(self.particles)
        weights = self.likelihood_func(preds, obs)
        self.__next = {
            'param': preds,
            'weight': weights,
        }
        return preds, weights