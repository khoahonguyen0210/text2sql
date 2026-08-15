"""Run the official evaluation.py as a script, with one reused asyncio loop.

Run from the evaluator folder. The evaluator's scoring code is not changed.
"""

import runpy
import sys

from text2sql.evaluation.official import patch_exec_eval

if __name__ == "__main__":
    sys.path.insert(0, "")
    patch_exec_eval()
    sys.argv = ["evaluation.py", *sys.argv[1:]]
    runpy.run_path("evaluation.py", run_name="__main__")
