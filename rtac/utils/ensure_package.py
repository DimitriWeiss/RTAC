import subprocess
import sys

def ensure_package(pkg_name, version):
    try:
        # Python 3.8+: Use importlib.metadata
        from importlib.metadata import version as get_version
    except ImportError:
        # Older versions
        from pkg_resources import get_distribution as get_version

    try:
        installed_version = get_version(pkg_name)
        if installed_version != version:
            raise ImportError
    except Exception:
        print(f"Installing {pkg_name}=={version} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", f"{pkg_name}=={version}"
        ])
