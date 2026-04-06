import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEVELOPMENT_SET = PROJECT_ROOT / "development_set"
INPUT_JSON = PROJECT_ROOT / "example_json" / "input.json"
OUTPUT_EXAMPLE = PROJECT_ROOT / "example_json" / "output_example.json"