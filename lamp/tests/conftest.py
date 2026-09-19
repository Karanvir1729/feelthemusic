"""Make `import ftm_client` resolve from lamp/ when pytest is run from the repository root."""
import sys
from pathlib import Path

_LAMP = str(Path(__file__).resolve().parents[1])
if _LAMP not in sys.path:
    sys.path.insert(0, _LAMP)
