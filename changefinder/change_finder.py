import statsmodels.api as sm
import numpy as np
import scipy as sp
from sdar import SDAR_1Dim


class _ChangeFinderAbstract(object):
    def _add_one(self, one, ts, size):
        ts.append(one)
        if len(ts) == size+1:
            ts.pop(0)

    def _smoothing(self, ts):
        return sum(ts)/float(len(ts))


class ChangeFinder(_ChangeFinderAbstract):
    def __init__(self, r=0.5, order=1, smooth=7):
        assert order > 0, "order must be 1 or more."
        assert smooth > 2, "term must be 3 or more."
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth/2.0))
        self._order = order
        self._r = r
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []
        self._sdar_first = SDAR_1Dim(r, self._order)
        self._sdar_second = SDAR_1Dim(r, self._order)

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._order:  # 第一段学習
            score, predict = self._sdar_first.update(x, self._ts)
            self._add_one(score, self._first_scores, self._smooth)
        self._add_one(x, self._ts, self._order)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target and len(self._smoothed_scores) == self._order:  # 第二段学習
            score, predict2 = self._sdar_second.update(second_target, self._smoothed_scores)
            self._add_one(score,
                          self._second_scores, self._smooth2)
        if second_target:
            self._add_one(second_target, self._smoothed_scores, self._order)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict


class ChangeFinderARIMA(_ChangeFinderAbstract):
    def __init__(self, term=30, smooth=7, order=(1, 0, 0)):
        assert smooth > 2, "term must be 3 or more."
        assert term > smooth, "term must be more than smooth"

        self._term = term
        self._smooth = smooth
        self._smooth2 = int(round(self._smooth/2.0))
        self._order = order
        self._ts = []
        self._first_scores = []
        self._smoothed_scores = []
        self._second_scores = []

    def _calc_outlier_score(self, ts, target):
        def outlier_score(residuals, x):
            m = residuals.mean()
            s = np.std(residuals, ddof=1)
            return -sp.stats.norm.logpdf(x, m, s)
        ts = np.array(ts)
        arima_model = sm.tsa.ARIMA(ts, self._order)
        result = arima_model.fit(disp=0)
        pred = result.forecast(1)[0][0]
        return outlier_score(result.resid, x=pred-target), pred

    def update(self, x):
        score = 0
        predict = x
        predict2 = 0
        if len(self._ts) == self._term:  # 第一段学習
            try:
                score, predict = self._calc_outlier_score(self._ts, x)
                self._add_one(score, self._first_scores, self._smooth)
            except Exception:
                self._add_one(x, self._ts, self._term)
                return 0, predict
        self._add_one(x, self._ts, self._term)
        second_target = None
        if len(self._first_scores) == self._smooth:  # 平滑化
            second_target = self._smoothing(self._first_scores)
        if second_target and len(self._smoothed_scores) == self._term:  # 第二段学習
            try:
                score, predict2 = self._calc_outlier_score(self._smoothed_scores, second_target)
                self._add_one(score,
                              self._second_scores, self._smooth2)
            except Exception:
                self._add_one(second_target, self._smoothed_scores, self._term)
                return 0, predict
        if second_target:
            self._add_one(second_target, self._smoothed_scores, self._term)
        if len(self._second_scores) == self._smooth2:
            return self._smoothing(self._second_scores), predict
        else:
            return 0.0, predict
