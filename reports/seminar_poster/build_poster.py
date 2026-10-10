"""Build the current poster; the historical builder is archived separately."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_poster_updated.py")), run_name="__main__")
