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
    def __init__(self, num_particle, n_dims, trans, observ, initial=None):
        self.num_particle = num_particle
        self.trans = trans
        self.observ = observ

        if initial is None:
            self.particles = np.zeros(num_particle, n_dims)
        else:
            self.particles = initial

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
        """リサンプリングは行わない"""
        preds = self.trans.predict(self.particles)
        weights = self.observ.likelihood(preds, obs)
        return preds, weights#, np.average(preds, weights=weights, axis=0)