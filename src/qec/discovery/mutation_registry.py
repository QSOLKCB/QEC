class MutationRegistry:

    def __init__(self):
        self._operators = []

    def register(self, operator):
        self._operators.append(operator)

    def operators(self):
        return list(self._operators)

    def scored(self, graph, eigenvector, context):
        scored = []

        for op in self._operators:
            score = float(op.score(graph, eigenvector, context))
            scored.append((score, op))

        # deterministic ordering
        scored.sort(key=lambda x: (-x[0], x[1].name))

        return [op for _, op in scored]
