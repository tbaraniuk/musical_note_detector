import numpy as np


class GaussianModel:
    def __init__(self, classes, num_features) -> None:
        self.num_classes = len(classes)
        self.alpha = np.ones(self.num_classes)
        self.num_features = num_features
        self.classes = np.array(list(classes))
        self.mu = np.zeros((self.num_classes, self.num_features))
        self.sigma = np.zeros((self.num_classes, self.num_features, self.num_features))
        
    def fit(self, X, y):
        for i in range(self.num_classes):
            idx = y == self.classes[i]
            X_c = X[idx]
            # self.mu[i] = self.mu[i] + (X_c - self.mu[i]) / (self.alpha[i] + len(X_c))
            self.mu[i] = np.mean(X_c, axis=0)
            self.sigma[i] = np.cov(X_c, rowvar=False) + 1e-3 * np.eye(self.num_features)
            
    def _log_likelihood(self, x, k):
        diff = x - self.mu[k]
        sigma_inv = np.linalg.inv(self.sigma[k])
        _, log_det = np.linalg.slogdet(self.sigma[k])
        return -0.5 * (diff @ sigma_inv @ diff + log_det)
    
    def predict(self, X):
        if X.ndim == 1:
            X = X[np.newaxis, :]
        predictions = []
        for x in X:
            log_posterior = np.zeros(self.num_classes)
            for k in range(self.num_classes):
                log_posterior[k] = self._log_likelihood(x, k) + np.log(self.alpha[k] / self.alpha.sum())
            
            predicted_idx = np.argmax(log_posterior)
            predictions.append(self.classes[predicted_idx])
            
        return np.array(predictions)
    
    def update(self, predicted_class):
        idx = np.where(self.classes == predicted_class)[0][0]
        self.alpha[idx] += 1
                
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
