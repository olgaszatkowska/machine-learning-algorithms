from typing import Literal
import numpy as np
from numpy.typing import NDArray


class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1):
        # xavier weight initialization
        # self.weights = np.random.randn(input_size) / np.sqrt(input_size)
        self.weights = (np.random.randn(input_size) / np.sqrt(input_size)) * 0.05
        self.bias = np.zeros(1)
        self.learning_rate = learning_rate

    def step(self, x: any) -> Literal[0, 1]:
        # binary step function
        return 1 if x > 0 else 0

    def backward(self, loss, inputs):
        self.weights += -self.learning_rate * loss * inputs
        self.bias += -self.learning_rate * loss

    def fit(self, X: NDArray, y: NDArray, epochs: int):
        for _ in range(epochs):
            for (inputs, target) in zip(X, y):
                prediction = self.step(np.dot(inputs, self.weights) + self.bias)
                loss = prediction - target
                self.backward(loss, inputs)

    def predict(self, X: NDArray) -> tuple[Literal[0, 1], float]:
        fn_value = (np.dot(X, self.weights) + self.bias)[0]
        return self.step(fn_value), fn_value


class PerceptronClassifier:
    def __init__(self, learning_rate: float = 0.1) -> None:
        self.perceptions = {}
        self.learning_rate = learning_rate

    def _generate_perceptrons(
        self, classes: list[int], input_size: int, learning_rate: float = 0.1
    ) -> dict[int, Perceptron]:
        return {
            target_class: Perceptron(input_size, learning_rate)
            for target_class in classes
        }

    def fit(self, X: NDArray, y: NDArray, epochs: int):
        self.perceptions = self._generate_perceptrons(
            np.unique(y).tolist(), X.shape[1], self.learning_rate
        )
        
        for target_class, perceptron in self.perceptions.items():
            y_targeted = self.transform_to_one_vs_all_labels(y, target_class)
            perceptron.fit(X, y_targeted, epochs)
    
    def predict(self, X: NDArray) -> int:
        predictions = ((target_class, p.predict(X)) for target_class, p in self.perceptions.items())
        confidence_score_mapping = {
            confidence_score: {"class": target_class, "prediction": prediction}
            for (target_class, (prediction, confidence_score)) in predictions
        }

        while len(confidence_score_mapping.keys()) > 1:
            highest_confidence_score = max(confidence_score_mapping.keys())
            highest_mapping_value = confidence_score_mapping.get(highest_confidence_score)
            if highest_mapping_value.get("prediction") == 1:
                return highest_mapping_value.get("class")
            del confidence_score_mapping[highest_confidence_score]

        return confidence_score_mapping.get(list(confidence_score_mapping.keys())[0]).get("class")

    @staticmethod
    def transform_to_one_vs_all_labels(y: NDArray, target_class: int) -> NDArray:
        one_vs_all_labels = np.zeros_like(y)
        one_vs_all_labels[y == target_class] = 1
        return one_vs_all_labels