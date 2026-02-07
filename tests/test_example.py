import os
import sys
import subprocess
import pytest

try:
    import torch
except Exception:
    torch = None

@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_run_pytorch_example(tmp_path):
    """Run the example script as a smoke test when torch is available."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run([sys.executable, "pytorch_zero_to_hero/examples/pytorch_sgd.py"], env=env, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0
    assert ("Saved loss plot" in proc.stdout) or (os.path.exists("pytorch_zero_to_hero/examples/artifacts/loss.png"))
