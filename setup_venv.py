import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional, Tuple

IS_WINDOWS = os.name == "nt"

def find_python_executable(version_spec: Optional[str]) -> str:
    """
    Find a python executable that matches version_spec.
    If version_spec is None, return the current sys.executable.
    version_spec examples: "3.8.10", "3.11", "3.10", "3"
    """
    if version_spec is None:
        return sys.executable

    version_spec = version_spec.strip()
    candidates = []
    if version_spec[0].isdigit():
        candidates.append(f"python{version_spec}")
        if "." in version_spec:
            parts = version_spec.split(".")
            major = parts[0]
            minor = parts[1] if len(parts) > 1 else None
            if minor:
                candidates.append(f"python{major}.{minor}")
            candidates.append(f"python{major}")
    else:
        candidates.append(version_spec)

    # Also try just "python" with the -V check fallback
    candidates.append("python")

    tried = []
    for name in candidates:
        path = shutil.which(name)
        tried.append(name)
        if not path:
            continue
        try:
            out = subprocess.check_output([path, "-c", "import sys;print(sys.version.split()[0])"], text=True, stderr=subprocess.DEVNULL)
            found_ver = out.strip()
            # Accept if full match or startswith version_spec (so "3.8" matches "3.8.10")
            if found_ver.startswith(version_spec):
                return path
            # Accept if version_spec is major only and matches found_ver
            if version_spec == found_ver.split(".")[0]:
                return path
            # Accept if version_spec has patch but python reports major.minor and they match
            if "." in version_spec:
                vs_parts = version_spec.split(".")
                found_parts = found_ver.split(".")
                if len(vs_parts) >= 2 and len(found_parts) >= 2 and vs_parts[0] == found_parts[0] and vs_parts[1] == found_parts[1]:
                    return path
        except Exception:
            continue

    # On Windows try the `py` launcher: if user requested patch version (3.8.10),
    # try py -3.8 first, then py -3.8.10 (py may accept only major.minor).
    if IS_WINDOWS and shutil.which("py"):
        # derive major.minor for py fallback
        try_versions = []
        if "." in version_spec:
            parts = version_spec.split(".")
            if len(parts) >= 2:
                try_versions.append(f"{parts[0]}.{parts[1]}")  # e.g. "3.8"
        try_versions.append(version_spec)  # e.g. "3.8.10" (py may or may not accept)
        # also try major only
        try_versions.append(version_spec.split(".")[0])

        for tv in try_versions:
            try:
                out = subprocess.check_output(["py", f"-{tv}", "-c", "import sys;print(sys.executable)"], text=True, stderr=subprocess.DEVNULL)
                return out.strip()
            except subprocess.CalledProcessError:
                continue

    raise RuntimeError(f"Requested Python {version_spec} not found on PATH (tried: {', '.join(tried)}). Install it or provide an exact path to the python executable.")

def read_numpy_pandas_versions(req_file: str = "requirements.txt") -> Tuple[Optional[str], Optional[str]]:
    """
    Parse requirements.txt for lines like:
      numpy==1.24.3
      pandas==2.0.3
    Returns (numpy_spec, pandas_spec) or (None, None)
    """
    numpy_spec = None
    pandas_spec = None
    if not os.path.exists(req_file):
        return numpy_spec, pandas_spec
    with open(req_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # handle comments after version
            if "#" in line:
                line = line.split("#", 1)[0].strip()
            low = line.lower()
            if low.startswith("numpy"):
                numpy_spec = line.split()[0]  # e.g. numpy==1.24.3
            elif low.startswith("pandas"):
                pandas_spec = line.split()[0]
    return numpy_spec, pandas_spec

def create_virtualenv(python_exe: str, env_name: str = "venv"):
    print(f"Creating virtual environment '{env_name}' using: {python_exe}")
    subprocess.check_call([python_exe, "-m", "venv", env_name])

def pip_path_for_env(env_name: str = "venv") -> str:
    if IS_WINDOWS:
        return os.path.join(env_name, "Scripts", "pip.exe")
    else:
        return os.path.join(env_name, "bin", "pip")

def python_path_for_env(env_name: str = "venv") -> str:
    if IS_WINDOWS:
        return os.path.join(env_name, "Scripts", "python.exe")
    else:
        return os.path.join(env_name, "bin", "python")

def install_requirements(env_name: str = "venv", use_binary_np_pd: bool = False):
    """
    Use the venv's python to run pip (python -m pip ...) to avoid Windows permission issues
    when pip tries to replace pip.exe while running.
    """
    py = python_path_for_env(env_name)
    if not os.path.exists(py):
        raise RuntimeError(f"python not found at {py}. Did venv creation fail?")

    # helper to run pip via "python -m pip"
    def run_pip(args, check=True):
        cmd = [py, "-m", "pip"] + args
        return subprocess.check_call(cmd) if check else subprocess.call(cmd)

    # Upgrade build tools first
    print("Upgrading pip, setuptools and wheel inside venv...")
    try:
        run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"])
    except subprocess.CalledProcessError as e:
        print("Failed to upgrade pip/setuptools/wheel via python -m pip:", e)
        raise

    # Optionally install numpy/pandas as binary wheels first to avoid build-from-source
    if use_binary_np_pd:
        numpy_spec, pandas_spec = read_numpy_pandas_versions()
        install_list = []
        if numpy_spec:
            install_list.append(numpy_spec)
        if pandas_spec:
            install_list.append(pandas_spec)
        if install_list:
            print("Installing binary wheels for (to avoid building from source):", install_list)
            try:
                run_pip(["install", "--only-binary=:all:"] + install_list)
            except subprocess.CalledProcessError:
                print("Warning: binary-only install failed for one or more packages; will continue to attempt full install.")

    # Finally install everything from requirements.txt (via python -m pip)
    print("Installing dependencies from requirements.txt...")
    run_pip(["install", "-r", "requirements.txt"])

def test_imports_with_env(env_name: str = "venv"):
    py = python_path_for_env(env_name)
    if not os.path.exists(py):
        print("Cannot find python in the venv to test imports.")
        return

    test_script = r'''
import sys
print("Using:", sys.executable)
errors = []
try:
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from collections import Counter
    from scipy.stats import kstest, kurtosis, skew
    from heapq import nlargest
    from sklearn import preprocessing
    import onnx
    import onnxruntime as ort
    import pickle
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("Success.")
'''

    print("Testing imports inside the venv...")
    try:
        subprocess.check_call([py, "-c", test_script])
    except subprocess.CalledProcessError:
        print("One or more imports failed. Re-run the venv python interactively for debugging:")
        print(f"  {py} -c \"import pandas, numpy; print(pandas.__version__, numpy.__version__)\"")

def main():
    parser = argparse.ArgumentParser(description="Create a venv using a specific Python version and install requirements.")
    parser.add_argument("--env", default="venv", help="Name/path of the virtual environment to create (default: venv)")
    parser.add_argument("--python", default="3.8.10", help="Python version to force (default: 3.8.10) or path to python executable.")
    parser.add_argument("--binary-numpy-pandas", action="store_true", help="Try installing numpy/pandas as binary wheels first to avoid building from source.")
    args = parser.parse_args()

    try:
        python_exe = find_python_executable(args.python)
    except RuntimeError as exc:
        print("ERROR:", exc)
        print("If you are on Windows, consider installing the desired Python from python.org and/or using the py launcher.")
        sys.exit(2)

    env = args.env

    # remove existing venv if present (optional safety - we won't remove unless user accepts)
    if os.path.exists(env):
        print(f"Virtual environment '{env}' already exists.")
        resp = input("Remove and recreate it? [y/N]: ").strip().lower()
        if resp == "y":
            if IS_WINDOWS:
                subprocess.check_call(["rmdir", "/S", "/Q", env], shell=True)
            else:
                shutil.rmtree(env)
        else:
            print("Using existing venv.")

    create_virtualenv(python_exe, env)
    install_requirements(env, use_binary_np_pd=args.binary_numpy_pandas)
    test_imports_with_env(env)

    print("\nTo activate the virtual environment, run:\n")
    if IS_WINDOWS:
        print(f"{env}\\Scripts\\Activate.ps1   # PowerShell")
        print(f"{env}\\Scripts\\activate.bat   # Command Prompt")
    else:
        print(f"source {env}/bin/activate")

if __name__ == "__main__":
    main()