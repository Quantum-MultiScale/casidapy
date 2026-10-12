"""Per-rank resident memory reporting for long MPI runs."""

from __future__ import annotations


def _proc_status_kb(key: str) -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return float(line.split()[1])
    except OSError:
        pass
    return 0.0


def report_memory(tag: str, comm=None) -> None:
    """Print current and peak RSS (max and sum over ranks) on rank 0.

    Collective when ``comm`` has more than one rank: every rank must call it.
    """
    rss = _proc_status_kb("VmRSS") / 2**20
    hwm = _proc_status_kb("VmHWM") / 2**20
    rank, size = 0, 1
    if comm is not None:
        rank = comm.Get_rank()
        size = comm.Get_size()
    if size > 1:
        from mpi4py import MPI

        rss_max = comm.allreduce(rss, op=MPI.MAX)
        rss_sum = comm.allreduce(rss, op=MPI.SUM)
        hwm_max = comm.allreduce(hwm, op=MPI.MAX)
        hwm_sum = comm.allreduce(hwm, op=MPI.SUM)
    else:
        rss_max = rss_sum = rss
        hwm_max = hwm_sum = hwm
    if rank == 0:
        print(
            f"[mem] {tag}: RSS max/rank {rss_max:.2f} GB, sum {rss_sum:.2f} GB; "
            f"peak max/rank {hwm_max:.2f} GB, sum {hwm_sum:.2f} GB",
            flush=True,
        )
