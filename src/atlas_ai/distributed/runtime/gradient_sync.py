class GradientSynchronizer:
    """
    Simulate distributed gradient aggregation.
    """

    def average_gradients(
        self,
        worker_gradients,
    ):
        """
        Average gradients across workers.
        """

        if not worker_gradients:
            return []

        num_workers = len(
            worker_gradients
        )

        num_gradients = len(
            worker_gradients[0]
        )

        averaged = []

        for gradient_index in range(
            num_gradients
        ):

            gradient_sum = 0.0

            for worker in worker_gradients:

                gradient_sum += worker[
                    gradient_index
                ]

            averaged.append(
                gradient_sum
                / num_workers
            )

        return averaged
