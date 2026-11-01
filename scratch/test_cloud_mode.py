import os
import sys
from pathlib import Path

# Ensure cloud mode and public DB are set BEFORE any module imports
os.environ["CLOUD_MODE"] = "1"
os.environ["DATABASE_URL"] = "sqlite:///data/public_reviews.db"

from streamlit.testing.v1 import AppTest

def test_cloud_mode():
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard.py"
    at = AppTest.from_file(str(dashboard_path), default_timeout=60)
    at.run()

    # The sidebar navigation radio options
    nav_options = list(at.sidebar.radio[0].options)
    print(f"Testing {len(nav_options)} views...")

    for opt in nav_options:
        at.sidebar.radio[0].set_value(opt).run()
        print(f"view: exceptions={len(at.exception)}")
        if at.exception:
            for ex in at.exception:
                print(f"  Exception in {opt}: {ex}")

    print("torch:", "torch" in sys.modules)
    print("transformers:", "transformers" in sys.modules)

if __name__ == "__main__":
    test_cloud_mode()
