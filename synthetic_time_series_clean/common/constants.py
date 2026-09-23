"""Project-wide constants for the rendered time-series format."""

SPECIAL_TOKENS = [
    "<static>",
    "</static>",
    "<event>",
    "</event>",
    "<comma>",
    "<newline>",
]

CLOSING_EVENT_TAG = "</event>"
EVENT_CSV_HEADER = "time,code,numeric_value"
VALID_EVAL_DATASETS = {"train", "tuning", "held_out"}
