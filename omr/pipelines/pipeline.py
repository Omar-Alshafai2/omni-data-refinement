"""
OMR Pipelines Module — Chainable fluent API for Dataset transformations.
"""

class Pipeline:
    """Enables fluent chaining of Dataset operations."""
    
    def __init__(self, dataset):
        self._dataset = dataset
        
    def health(self):
        self._dataset.health()
        return self
        
    def clean(self):
        self._dataset.clean()
        return self
        
    def analyze(self):
        self._dataset.analyze()
        return self
        
    def validate(self, schema):
        self._dataset.validate(schema)
        return self
        
    def export(self):
        return self._dataset.export()
