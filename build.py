import argparse
import subprocess
import sys
import os
import shutil

def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        print(f"Loading environment variables from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def main(run_after_build, no_cache):
    # Load environment variables first
    load_env_file()
    
    build_dir = "build"

    if no_cache and os.path.exists(build_dir):
        print(f"Removing existing build directory '{build_dir}' due to --no_cache...")
        shutil.rmtree(build_dir)

    os.makedirs(build_dir, exist_ok=True)

    cmake_configure_cmd = [
        "cmake",
        "-S", ".",
        "-B", build_dir,
        "-G", "Visual Studio 17 2022",
        "-DCMAKE_BUILD_TYPE=Release"
    ]

    print("Configuring project...")
    subprocess.check_call(cmake_configure_cmd)

    cmake_build_cmd = [
        "cmake",
        "--build", build_dir,
        "--config", "Release"
    ]

    print("Building project...")
    subprocess.check_call(cmake_build_cmd)

    if run_after_build:
        exe_path = os.path.join(build_dir, "Release", "ChessEngine.exe")
        if not os.path.exists(exe_path):
            exe_path = os.path.join(build_dir, "ChessEngine.exe")
        
        if os.path.exists(exe_path):
            print(f"Running {exe_path} ...")
            subprocess.run(exe_path)
        else:
            print(f"Executable not found. Searched at: {exe_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and optionally run ChessEngine.")
    parser.add_argument("--run", action="store_true", help="Run executable after build")
    parser.add_argument("--no_cache", action="store_true", help="Force rebuild by deleting build directory")
    args = parser.parse_args()

    try:
        main(args.run, args.no_cache)
    except subprocess.CalledProcessError as e:
        print(f"Build error: {e}")
        sys.exit(1)