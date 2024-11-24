from sklearn.linear_model import LogisticRegression

class LogisticRegressionWeatherForecast:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.condition_mapping = {"Sunny": 0, "Rainy": 1, "Cloudy": 2, "Uncertain": 3}
        self.reverse_mapping = {v: k for k, v in self.condition_mapping.items()}

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        predictions = self.model.predict(X)
        return [self.reverse_mapping[p] for p in predictions]