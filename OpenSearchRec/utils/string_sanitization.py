from enum import Enum
import re


def sanitize_url_string_component(url_string_component, max_length=256):
    if isinstance(url_string_component, Enum):
        url_string_component = url_string_component.value

    if type(url_string_component) in [int, float]:
        url_string_component = str(url_string_component)

    if not type(url_string_component) == str:
        raise ValueError(f"url_string_component must be a string for value={url_string_component}")

    if not len(url_string_component) > 0:
        raise ValueError(f"url_string_component length must be greater than 0 for value={url_string_component}")

    if not len(url_string_component) <= max_length:
        raise ValueError(f"Exception: url_string_component length must be less than or equal to {url_string_component} "
                         f"for value={url_string_component}")

    regex = "^[a-zA-Z0-9_: \-]+$"
    if not re.match(regex, url_string_component):
        raise ValueError(f"Exception: value must match regex '{regex}' for value={url_string_component}")

    return url_string_component
