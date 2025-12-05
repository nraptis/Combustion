# data_label_collection.py
from __future__ import annotations

from typing import Any, Dict, List, Iterable

from labels.data_label import DataLabel

class DataLabelCollection:
    """
    A simple list of DataLabel objects.

    Multiple labels may share the same name. Each DataLabel typically
    represents one instance (one PixelBag region).
    """

    def __init__(self, labels: List[DataLabel] | None = None) -> None:
        self.labels: List[DataLabel] = labels or []

    # --------------------------------------------------
    # Basic operations
    # --------------------------------------------------
    def add_label(self, label: DataLabel) -> None:
        """
        Append a label. Multiple labels may have the same name.
        """
        self.labels.append(label)

    def remove_label(self, label: DataLabel) -> None:
        """
        Remove this exact label object from the collection, if present.
        Does nothing if the label is not in the collection.
        """
        try:
            self.labels.remove(label)
        except ValueError:
            # label not in list; ignore
            pass

    def get_labels_by_name(self, name: str) -> List[DataLabel]:
        """
        Return all labels whose name matches the given name.
        May return an empty list.
        """
        return [lbl for lbl in self.labels if lbl.name == name]

    def first_label(self, name: str) -> DataLabel | None:
        """
        Convenience: return the first label with this name, or None.
        """
        for lbl in self.labels:
            if lbl.name == name:
                return lbl
        return None

    # --------------------------------------------------
    # JSON serialization (array only)
    # --------------------------------------------------
    def to_json(self) -> List[Dict[str, Any]]:
        """
        Return a JSON-compatible list of label dicts, no wrapper.
        """
        return [label.to_json() for label in self.labels]

    @staticmethod
    def from_json(data: List[Dict[str, Any]]) -> "DataLabelCollection":
        """
        Parse a list of label objects.
        """
        labels = [DataLabel.from_json(item) for item in data]
        return DataLabelCollection(labels=labels)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def __len__(self):
        return len(self.labels)

    def __iter__(self) -> Iterable[DataLabel]:
        return iter(self.labels)

    def _sorted_labels(self) -> List[DataLabel]:
        """
        Return labels sorted by:
          1) name (alphabetical)
          2) median.y  (ascending)
          3) median.x  (ascending)

        Uses PixelBag.summary().median so we don't have to look at stripes.
        Empty bags get a large sentinel so they sort last within a name.
        """
        if not self.labels:
            return []

        def sort_key(label: DataLabel):
            name = label.name
            summary = label.pixel_bag.summary()
            median = summary.get("median")

            if median is None:
                # Empty bag: push to the end for that name
                my = 10**9
                mx = 10**9
            else:
                mx, my = median  # median is (x, y)

            return (name, my, mx)

        return sorted(self.labels, key=sort_key)

    def __repr__(self) -> str:
        """
        Compact summary of the collection plus one-line summary per label, e.g.:

            DataLabelCollection(count=2):
                DataLabel(name="basal", bag=PixelBag(...))
                DataLabel(name="lymph", bag=PixelBag(...))
        """
        count = len(self.labels)
        if count == 0:
            return "DataLabelCollection(count=0)"

        lines: List[str] = [f"DataLabelCollection(count={count}):"]
        for label in self._sorted_labels():
            lines.append(f"    {label!r}")
        return "\n".join(lines)
    