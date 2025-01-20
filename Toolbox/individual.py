class Individual:
    def __init__(self, dictionary):
        for key in dictionary:
            setattr(self, key, dictionary[key])

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return f"{vars(self)}"