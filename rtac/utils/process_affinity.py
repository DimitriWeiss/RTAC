import psutil
import subprocess


def set_affinity_recursive(proc: subprocess.Popen, core: int) -> None:
    """
    Bind the given process to the given core.

    Parameters
    ----------
    proc : subprocess.Process
        Process that is to be bound.
    core : int
        Core that the process is to be bound to.

    Returns
    -------
    None
    """
    try:
        proc = psutil.Process(proc.pid)
        proc.cpu_affinity([core])
        for child in proc.children(recursive=True):
            child.cpu_affinity([core])
    except psutil.NoSuchProcess:
        pass
