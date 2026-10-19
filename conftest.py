"""Make the repository root importable from tests.

The tests import the application packages directly (`core`, `retrieval`, …),
which only works when the repository root is on `sys.path`. Running
`python -m pytest` happens to put the current directory there, but bare
`pytest` — which is what CI runs — does not, and the suite then fails at
collection with `ModuleNotFoundError: No module named 'core'`.

Inserting the root here makes both invocations behave the same, without asking
every test module to do it for itself.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
