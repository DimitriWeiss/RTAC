import psutil


def set_affinity_recursive(proc, core):
    try:
        proc = psutil.Process(proc.pid)
        proc.cpu_affinity([core])
        for child in proc.children(recursive=True):
            child.cpu_affinity([core])
    except psutil.NoSuchProcess:
        pass
