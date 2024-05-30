from model import Perceptron
from numpy.typing import NDArray


def one_vs_all(p0: Perceptron, p1: Perceptron, p2: Perceptron, sample: NDArray) -> int:
    perceptrons = [p0, p1, p2]
    predictions = ((i, p.predict(sample)) for i, p in enumerate(perceptrons))
    prediction_mapping = {
        confidence_score: {"id": i, "prediction": prediction}
        for (i, (prediction, confidence_score)) in predictions
    }

    while len(prediction_mapping.keys()) > 1:
        highest_confidence_score = max(prediction_mapping.keys())
        highest_mapping_value = prediction_mapping.get(highest_confidence_score)
        if highest_mapping_value.get("prediction") == 1:
            return highest_mapping_value.get("id")
        del prediction_mapping[highest_confidence_score]

    return prediction_mapping.get(list(prediction_mapping.keys())[0]).get("id")
