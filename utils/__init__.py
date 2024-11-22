from .prediction_utils import input_array
#from .ml_utils import
from typing import Any
import orjson
from fastapi import Response


class CustomORJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        assert orjson is not None, "orjson must be installed"
        return orjson.dumps(content, option=orjson.OPT_INDENT_2)

def formatter(string):
    string = string.split('T')[1].split('.')[0]
    hours, minutes, seconds = map(int, string.split(':'))
    # Calculate total minutes since start of the day
    total_minutes = hours * 60 + minutes

    return total_minutes