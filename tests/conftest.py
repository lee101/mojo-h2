from __future__ import annotations

import os
import subprocess

import pytest


@pytest.fixture(scope="session", autouse=True)
def build_library():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(
        ["bash", os.path.join(root, "build", "build.sh")],
        cwd=root,
        check=True,
    )
