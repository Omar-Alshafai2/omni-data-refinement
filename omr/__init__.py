"""
OMR - Omni Data Refinement v0.3.0
The standard framework for dataset quality, validation, profiling, monitoring, and reliability.
"""

# Core
from .core.dataset import Dataset
from .core.metadata import DatasetMetadata

# Main Engines & Models
from .health.engine import HealthReport, Issue
from .profiling.engine import ProfilingReport, ColumnProfile
from .validation.engine import ValidationReport, ValidationIssue
from .statistics.engine import StatisticsReport, StatisticalFinding
from .drift.engine import DriftReport, DriftResult
from .versioning.system import VersionSnapshot

# Ops
from .monitoring.monitor import Monitor, Alert
from .pipelines.pipeline import Pipeline
from .plugins.registry import register_plugin, get_plugin

# Schemas
from .schemas import types as schemas

# Backward Compatibility (Deprecation notice could be added here in the future)
OMRAssistant = Dataset


__version__ = "0.3.0"
__all__ = [
    # Primary interface
    "Dataset",
    "Monitor",
    "Pipeline",
    "schemas",
    "register_plugin",
    "get_plugin",
    
    # Reports & Models
    "DatasetMetadata",
    "HealthReport", "Issue",
    "ProfilingReport", "ColumnProfile",
    "ValidationReport", "ValidationIssue",
    "StatisticsReport", "StatisticalFinding",
    "DriftReport", "DriftResult",
    "VersionSnapshot",
    "Alert",
    
    # Aliases
    "OMRAssistant"
]
