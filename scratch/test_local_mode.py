import os
import sys
from pathlib import Path

# Unset CLOUD_MODE, use default DB
os.environ.pop("CLOUD_MODE", None)
os.environ.pop("DATABASE_URL", None)

from streamlit.testing.v1 import AppTest

def test_local_mode():
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard.py"
    at = AppTest.from_file(str(dashboard_path), default_timeout=60)
    at.run()

    nav_options = list(at.sidebar.radio[0].options)
    print(f"Testing {len(nav_options)} views in local mode...")

    for opt in nav_options:
        at.sidebar.radio[0].set_value(opt).run()
        print(f"view: exceptions={len(at.exception)}")
        if at.exception:
            for ex in at.exception:
                print(f"  Exception in {opt}: {ex}")

if __name__ == "__main__":
    test_local_mode()
