from enum import Enum
import re
from pydantic import BaseModel, Extra


class DatabaseType(str, Enum):
    elasticsearch = "elasticsearch"
    opensearch = "opensearch"


class TextMatchType(str, Enum):
    cross_fields = "cross_fields"
    best_fields = "best_fields"


class FieldTypeNamesAndPrefixes(str, Enum):
    text_fields = "text_"
    categorical_fields = "categorical_"
    numeric_fields = "numeric_"
    date_fields = "date_"
    geolocation_fields = "geolocation_"
    embedding_fields = "embedding_"


class OperationRefreshConfig(str, Enum):
    true = "true"
    false = "false"
    wait_for = "wait_for"


class ScalarBoostModifierFunctions(str, Enum):
    none = "none"
    log1p = "log1p"
    log2p = "log2p"
    sqrt = "sqrt"
    square = "square"


class BoostDecayFunctions(str, Enum):
    exponential = "exponential"
    linear = "linear"
    gaussian = "gaussian"


class TimeUnits(str, Enum):
    seconds = "seconds"
    minutes = "minutes"
    hours = "hours"
    days = "days"


class EmbeddingComparisonMetric(str, Enum):
    cosine_similarity = "cosine_similarity"
    dot_product = "dot_product"
    l2_norm = "l2_norm"
    l1_norm = "l1_norm"


class DistanceUnit(str, Enum):
    meters = "meters"
    kilometers = "kilometers"


class GeolocationField(BaseModel, extra=Extra.forbid):
    latitude: float
    longitude: float


def raise_exception_if_field_name_not_valid(field_name, max_field_name_length=256):
    if not type(field_name) == str:
        raise ValueError(f"value must be a string for value={field_name}")
    if not len(field_name) > 0:
        raise ValueError(f"value length must be greater than 0 for value={field_name}")
    if not len(field_name) <= max_field_name_length:
        raise ValueError(f"Exception: value length must be less than or equal to {max_field_name_length} "
                         f"for value={field_name}")
    regex = "^[a-zA-Z0-9_: \-]+$"
    if not re.match(regex, field_name):
        raise ValueError(f"Exception: value must match regex '{regex}' for value={field_name}")


def raise_exception_if_geolocation_field_value_not_valid(field_value):
    if not type(field_value) == dict:
        raise ValueError("geolocation fields must be of type Dict[str, float]")
    if len(field_value.keys()) != 2:
        raise ValueError('geolocation fields must be formatted as {"lat": 0, "long": 0}')
    if "lat" not in field_value:
        raise ValueError('geolocation fields must be formatted as {"lat": 0, "long": 0}')
    if "long" not in field_value:
        raise ValueError('geolocation fields must be formatted as {"lat": 0, "long": 0}')
    if (type(field_value["lat"]) not in [int, float]) or (type(field_value["long"]) not in [int, float]):
        raise ValueError('geolocation fields must be formatted as {"lat": 0, "long": 0} '
                         'and have lat/long values of type float')
