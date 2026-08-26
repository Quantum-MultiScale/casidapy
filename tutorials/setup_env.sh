#!/usr/bin/env bash
# Source this BEFORE starting Jupyter / VS Code Python kernel on Amarel.
#
#   source /projectsn/mp1009_1/am4655/casidapy/tutorials/setup_env.sh
#   jupyter lab   # or: Cursor — reload window / restart kernel after sourcing in the terminal used to launch the IDE
#
# Community Python 3.10 links against libffi.so.7; RHEL9 only ships .so.8.
set -euo pipefail

module use /projects/community-old/modulefiles 2>/dev/null || true
module load libffi/3.3-gc563 2>/dev/null || true

export OPAL_PREFIX="${OPAL_PREFIX:-/projects/community-old/gcc/11.2/openmpi/4.1.6/ez82}"
_LIBFFI="${LIBFFI_ROOT:-/projects/community-old/libffi/3.3/gc563/lib64}"

export LD_LIBRARY_PATH="${_LIBFFI}:${OPAL_PREFIX}/lib:/projects/community-old/openssl-3.0.14/lib:/projects/community-old/python/3.10/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMPI_MCA_btl=^openib
export PYTHONPATH="/projectsn/mp1009_1/am4655/casidapy${PYTHONPATH:+:$PYTHONPATH}"

# Optional: activate stddft venv if present
if [[ -f /projectsn/mp1009_1/am4655/stddft/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /projectsn/mp1009_1/am4655/stddft/bin/activate
fi

echo "[casidapy tutorials] LD_LIBRARY_PATH starts with: ${_LIBFFI}"
python -c "import ctypes; print('[casidapy tutorials] ctypes OK')" 2>/dev/null \
  || echo "[casidapy tutorials] WARNING: ctypes still broken — check module load libffi/3.3-gc563"
# pylibxc is linked into the stddft venv (do NOT module load pylibxc — gcc/10.2 deps are broken on el9)
python -c "import pylibxc; print('[casidapy tutorials] pylibxc', getattr(pylibxc, '__version__', 'OK'))" 2>/dev/null \
  || echo "[casidapy tutorials] WARNING: pylibxc missing from stddft site-packages"
