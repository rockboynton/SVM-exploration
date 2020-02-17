import numpy as np


class SVM:
    """
    Implements a suport vector machine that uses sub-gradient decent
    """

    def __init__(self, iter, strictness):
        """
        iter -- the number of iterations to run
        strictness -- the factor that determines how strict the svm will be
        """
        self.iter = iter
        self.strictness = strictness

    def fit(self, X, y):
        """
        Sets the weights based on the input
        X -- Matrix of observations and features
        y -- Array of categories, needs to be either -1 or 1
        """
        w = np.zeros((X.shape[1] + 1))
        _X = np.hstack((X, np.ones((X.shape[0], 1))))

        t = 1
        for iter in range(0, self.iter):
            for i in np.random.permutation(_X.shape[0]):
                step = 1 / (t * self.strictness) if self.strictness > 0 else .1
                x = _X[i]
                if y[i] * np.sum(x * w) < 1:
                    w = w - step * (self.strictness * w - y[i] * x)
                else:
                    w = w - step * self.strictness * w
                t += 1
        return w

    def predict(self, X, w):
        _X = np.hstack((X, np.ones((X.shape[0], 1))))
        p = np.sum(_X * w, axis=1)
        return p / abs(p)

    def line(self, X, w, independent_index, dependent_index, margin=0):
        independent = X[:, independent_index]

        divBy = -w[dependent_index]
        multBy = w[independent_index]
        addBy = 0
        for i in range(0, len(w) - 1):
            if i != independent_index and i != dependent_index:
                addBy += np.mean(X[:, i]) * w[i]
        addBy += w[len(w) - 1] + margin

        i1 = np.min(independent)
        i2 = np.max(independent)

        return np.array([[i1, (i1 * multBy + addBy) / divBy],
                         [i2, (i2 * multBy + addBy) / divBy]])
