import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROBOT_DIR = Path(os.environ.get("FTM_ROBOT_DIR",
                                Path.home() / "lelamp-hackathon-2026/static/robots/lelamp_v1/pi5_feetech_r1"))
CALIBRATION = Path(os.environ.get("FTM_CALIBRATION", Path.home() / "lelamp-hackathon-2026/lelamp.json"))

needs_robot = pytest.mark.skipif(not (ROBOT_DIR / "robot.urdf").exists(),
                                 reason="vendor robot description not found (set FTM_ROBOT_DIR)")
