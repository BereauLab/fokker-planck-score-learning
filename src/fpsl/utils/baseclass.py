from abc import ABC
from dataclasses import fields, _MISSING_TYPE


class DefaultDataClass(ABC):
    def __post_init__(self) -> None:
        # Loop through the fields to fill default values
        for field in fields(self):
            if (
                not isinstance(field.default, _MISSING_TYPE)
                and getattr(self, field.name) is None
            ):
                setattr(self, field.name, field.default)
