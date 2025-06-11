import os
import psutil


def is_file_open(filepath):
    """Check if any process is using the file."""
    for proc in psutil.process_iter(['open_files']):
        try:
            files = proc.info['open_files']
            if files:
                for f in files:
                    if f.path == filepath:
                        return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def remove_fuse_hidden_files(directory):
    for fname in os.listdir(directory):
        if fname.startswith('.fuse_hidden'):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath) and not is_file_open(fpath):
                os.remove(fpath)
