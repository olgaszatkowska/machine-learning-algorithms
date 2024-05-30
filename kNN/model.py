import math
from numpy.typing import NDArray
from collections import Counter


class kNNClassifier:
    def __init__(self, nearest_neighbors_max_count: int) -> None:
        self.max_count = nearest_neighbors_max_count
        self.X_train = None
        self.y_train = None

    def fit(self, X: NDArray, y: NDArray):
        self.X_train = X
        self.y_train = y

    def predict(self, X) -> int:
        neighbors = self._get_nearest_neighbors(X)
        max_class = self.get_max_class(neighbors)

        return max_class

    def _get_nearest_neighbors(self, X) -> list[tuple[NDArray, int, float]]:
        nearest_neighbors = []

        for (attributes, target) in zip(self.X_train, self.y_train):
            nearest_neighbors = self._add_or_decline(
                nearest_neighbors, attributes, int(target), X
            )

        return nearest_neighbors

    def _add_or_decline(
        self,
        nearest_neighbors: list[tuple[NDArray, int, float]],
        candidate_attributes: NDArray,
        candidate_class: int,
        X: NDArray,
    ) -> list[tuple[NDArray, int, float]]:
        distance_to_candidate = self.distance(candidate_attributes, X)

        is_empty = len(nearest_neighbors) == 0

        if is_empty:
            nearest_neighbors.append(
                (candidate_attributes, candidate_class, distance_to_candidate)
            )
            return nearest_neighbors

        _, _, distance_to_last = nearest_neighbors[-1]
        is_full = len(nearest_neighbors) == self.max_count

        if distance_to_last < distance_to_candidate and not is_full:
            nearest_neighbors.append(
                (candidate_attributes, candidate_class, distance_to_candidate)
            )
            return nearest_neighbors

        if distance_to_last < distance_to_candidate and is_full:
            return nearest_neighbors

        for i, data in enumerate(nearest_neighbors):
            _, _, distance_to_neighbor = data
            if distance_to_candidate < distance_to_neighbor:
                nearest_neighbors.insert(
                    i, (candidate_attributes, candidate_class, distance_to_candidate)
                )
                return (
                    nearest_neighbors[: self.max_count]
                    if is_full
                    else nearest_neighbors
                )

        return nearest_neighbors

    @staticmethod
    def distance(point1: NDArray, point2: NDArray) -> int:
        squared_diff_sum = sum((x2 - x1) ** 2 for x1, x2 in zip(point1, point2))
        return math.sqrt(squared_diff_sum)

    @staticmethod
    def get_max_class(neighbors: list[tuple[NDArray, int, float]]) -> int:
        target_classes = [target for _, target, _ in neighbors]
        counter = Counter(target_classes)

        most_common = counter.most_common(1)
        return most_common[0][0]
