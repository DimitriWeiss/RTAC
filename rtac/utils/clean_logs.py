import os
import psutil


def is_file_open(filepath: str) -> bool:
    """
    Check if any process is using the file.

    Parameters
    ----------
    filepath : str
        Path to the file we want to know if it is open.

    Returns
    -------
    bool
        True if file is open, False if not.
    """
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


def remove_fuse_hidden_files(directory: str) -> None:
    """
    Removes hidden fuse files that may clutter the log directory.

    Parameters
    ----------
    directory : str
        Path to the directory to declutter from fuse files.

    Returns
    -------
    None
    """
    for fname in os.listdir(directory):
        if fname.startswith('.fuse_hidden'):
            fpath = os.path.join(directory, fname)
            if os.path.isfile(fpath) and not is_file_open(fpath):
                os.remove(fpath)
