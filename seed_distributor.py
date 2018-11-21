import random
import numpy as np
import tensorflow as tf


class SeedDistributor:
    def __init__(self, random_seed):
        self.random_seed = random_seed
        if self.random_seed < 0:
            self.random_seed = random.randint(1, 1 << 32)

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

        self.__active_seed__ = self.random_seed

    def register_seed(self):
        self.__active_seed__ += 1

        return self.__active_seed__
