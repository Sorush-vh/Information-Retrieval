import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        self._validate(actual, predicted)
        precisions = []
        for a, p in zip(actual, predicted):
            if len(p) == 0:
                precisions.append(0)
                continue

            tp = len(set(a) & set(p))
            precisions.append(tp / len(p))

        return sum(precisions) / len(precisions)

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        self._validate(actual, predicted)

        recalls = []
        for a, p in zip(actual, predicted):
            if len(a) == 0:
                recalls.append(0)
                continue

            tp = len(set(a) & set(p))
            recalls.append(tp / len(a))

        return sum(recalls) / len(recalls)

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)

        if precision + recall == 0:
            return 0

        return 2 * precision * recall / (precision + recall)


    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        if not actual:
            return 0

        score = 0
        hits = 0

        for i, doc in enumerate(predicted, start=1):
            if doc in actual:
                hits += 1
                score += hits / i

        return score / len(actual)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        self._validate(actual, predicted)

        aps = [
            self._average_precision_single(a, p)
            for a, p in zip(actual, predicted)
        ]

        return sum(aps) / len(aps)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)


    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        dcg = 0

        for i, doc in enumerate(predicted, start=1):
            rel = 1 if doc in actual else 0
            dcg += rel / math.log2(i + 1)

        return dcg

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        self._validate(actual, predicted)

        dcgs = [
            self._dcg_single(a, p)
            for a, p in zip(actual, predicted)
        ]
        return sum(dcgs) / len(dcgs)

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        self._validate(actual, predicted)

        scores = []

        for a, p in zip(actual, predicted):
            dcg = self._dcg_single(a, p)

            ideal = a[:]
            idcg = self._dcg_single(a, ideal)

            if idcg == 0:
                scores.append(0)
            else:
                scores.append(dcg / idcg)

        return sum(scores) / len(scores)

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        self._validate(actual, predicted)

        rr = []

        for a, p in zip(actual, predicted):
            rank = 0

            for i, doc in enumerate(p, start=1):
                if doc in a:
                    rank = i
                    break

            if rank == 0:
                rr.append(0)
            else:
                rr.append(1 / rank)

        return sum(rr) / len(rr)

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        return self.cacluate_RR(actual, predicted)
        
    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)

        self.print_evaluation(
            precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr
        )

        self.log_evaluation(
            precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr
        )
