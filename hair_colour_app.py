"""
Compatibility wrapper for hair_color_app.py (supporting British/Commonwealth spelling).
Runs hair_color_app.py directly.
"""
import runpy
from pathlib import Path

target = Path(__file__).resolve().parent / "hair_color_app.py"
runpy.run_path(str(target), run_name="__main__")
