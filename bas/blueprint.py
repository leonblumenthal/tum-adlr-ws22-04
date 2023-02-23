import numpy as np


class Blueprint:
    """Blueprint of a BAS environment specifying all static obstacles,
    including the world size."""

    def __init__(self, world_size: np.ndarray):
        """Initialze the blueprint.

        Args:
            world_size: World size of the BAS environment.
        """

        assert world_size.shape == (2,)
        self.world_size = world_size.astype(float)
