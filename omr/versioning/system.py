"""
OMR Versioning Module — Tracks datasets and transformation histories.
"""
from dataclasses import dataclass
import pandas as pd
from ..core.metadata import DatasetMetadata


@dataclass
class VersionSnapshot:
    version_id: int
    name: str
    description: str
    df: pd.DataFrame
    metadata: DatasetMetadata


class VersioningSystem:
    def __init__(self):
        self._versions: list[VersionSnapshot] = []
        self._counter = 0

    def snapshot(self, df: pd.DataFrame, metadata: DatasetMetadata, 
                 name: str = "", description: str = "") -> int:
        self._counter += 1
        if not name:
            name = f"v{self._counter}"
            
        import copy
        # Deep copy the metadata to preserve transformation history at this point in time
        snap_meta = copy.deepcopy(metadata)
            
        version = VersionSnapshot(
            version_id=self._counter,
            name=name,
            description=description,
            df=df.copy(),
            metadata=snap_meta
        )
        self._versions.append(version)
        return self._counter

    def rollback(self, version_id: int) -> tuple[pd.DataFrame, DatasetMetadata]:
        for v in self._versions:
            if v.version_id == version_id:
                import copy
                return v.df.copy(), copy.deepcopy(v.metadata)
        raise ValueError(f"Version {version_id} not found.")

    def get_history(self) -> list[VersionSnapshot]:
        return self._versions
