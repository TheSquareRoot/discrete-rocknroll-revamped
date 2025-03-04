class Builder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dicts into DistributionBuilder instances
                setattr(self, key, Builder(**value))
            else:
                setattr(self, key, value)

    def generate(self):
        pass