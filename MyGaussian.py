import numpy as np

class MyGaussianNB(object):
    def __init__(self):
        self.n_features = None
        self.labels = None
        self.label_probs = {}
        self.label_mus = {}
        self.label_sigmas = {}

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)

        if(x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]):
            raise ValueError("训练数据有误")

        self.n_features = x.shape[1]

        labels = np.unique(y)
        self.labels = list(labels)

        for label in labels:
            self.label_probs[label] = (y == label).mean()
            x_label = x[y == label]
            self.label_mus[label] = x_label.mean(axis=0)
            self.label_sigmas[label] = x_label.std(axis=0)

    def _gaussian(self, x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-0.5 * ((x-mu)/sigma) ** 2)
        
    def predict(self, x):
        x = np.array(x)

        if(x.ndim != 2 or x.shape[1] != self.n_features):
            raise ValueError("训练数据有误")
        
        results = []
        
        for x_ in x:
            probs = []
            for label in self.labels:
                prob = self._gaussian(x=x_,mu=self.label_mus[label],sigma=self.label_sigmas[label])
                prob = np.prod(prob) * self.label_probs[label]
                probs.append(prob)
            results.append(self.labels[np.argmax(probs)])
        
        return np.array(results)

