class KVCache:
    """
    Simplified key-value cache for
    transformer inference.
    """

    def __init__(self):

        self.keys = []

        self.values = []

    def add(
        self,
        key,
        value,
    ):

        self.keys.append(key)

        self.values.append(value)

    def get_keys(self):

        return self.keys

    def get_values(self):

        return self.values

    def size(self):

        return len(self.keys)

    def clear(self):

        self.keys = []

        self.values = []
