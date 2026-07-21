#!/bin/bash

#SBATCH --job-name=Casida_TDDFT
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --output=casida_%j.out
#SBATCH --error=casida_%j.err

#
# Step 2: Run Casida TDDFT (MPI, CPU)
#
# Runs ``casidapy.run_casida_parallel_generic``: builds the KS Casida active space from
# ``--n-occ``, ``--n-unocc``, and optional ``--n-total-occ`` (HOMO count = LUMO index),
# then solves dense or matrix-free (``--matrix-free`` + ``--solver-method``).
#
# Requires SCF outputs: density, wavefunctions, eigenvalues, occupations (see run_scf.sh).
#
# Usage:
#   # Minimal (auto-detect orbital counts):
#   sbatch scripts/run_casida.sh \
#       --workdir examples/AG4_PARALLEL \
#       --atoms ag4.vasp \
#       --density rho_scf_ag4.xsf \
#       --psi psi_ag4.npy \
#       --eigs eig_ag4.npy \
#       --occs occs_ag4.npy \
#       --use-uspp --uspp-map "Ag:ag_pbe_v1.4.uspp.F.UPF" \
#       --xc PBE
#
#   # Full control:
#   sbatch scripts/run_casida.sh \
#       --workdir examples/AG4_PARALLEL \
#       --atoms ag4.vasp \
#       --density rho_scf_ag4.xsf \
#       --psi psi_ag4.npy \
#       --eigs eig_ag4.npy \
#       --occs occs_ag4.npy \
#       --use-uspp --uspp-map "Ag:ag_pbe_v1.4.uspp.F.UPF" \
#       --xc PBE \
#       --n-occ 38 --n-unocc 32 --n-states 50 \
#       --n-total-occ 38 \
#       --output-prefix casida_ag4 \
#       --matrix-free --solver-method lobpcg \
#       --plot
#
#   # Without USPP (norm-conserving only):
#   sbatch scripts/run_casida.sh \
#       --workdir examples/AG4_PARALLEL \
#       --atoms ag4.vasp \
#       --density rho_scf_ag4.xsf \
#       --psi psi_ag4.npy \
#       --eigs eig_ag4.npy \
#       --occs occs_ag4.npy \
#       --pseudo-map "Ag:Ag_ONCV_PBE-1.2.upf" \
#       --xc PBE
#
#   # Triplet excitations (PBE, requires pylibxc):
#   sbatch scripts/run_casida.sh \
#       --workdir examples/AG4_PARALLEL \
#       --atoms ag4.vasp \
#       --density rho_scf_ag4.xsf \
#       --psi psi_ag4.npy \
#       --eigs eig_ag4.npy \
#       --occs occs_ag4.npy \
#       --pseudo-map "Ag:Ag_ONCV_PBE-1.2.upf" \
#       --xc PBE \
#       --spin-state triplet \
#       --output-prefix casida_ag4_triplet
#
# All arguments after the SBATCH directives are passed directly to
# run_casida_parallel_generic.py via mpirun/srun.
#
# Python: set PYTHON_EXE before running, or activate a venv — see below.
#

set -e

# ============================================================
# Environment (adjust modules for your cluster)
# ============================================================
module purge 2>/dev/null || true
if [ -d /projects/community-old/modulefiles ]; then
    module use /projects/community-old/modulefiles
fi
# Examples:  module load eDFTpy
#          or: module load gcc/10.2.0/openmpi/4.0.5-bz186
if [ -f "${STDDFT_VENV:-/cache/home/am4655/edftpy_proj/edft}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${STDDFT_VENV:-/cache/home/am4655/edftpy_proj/edft}/bin/activate"
fi
export PATH PYTHONPATH VIRTUAL_ENV

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    STDDFT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    STDDFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SRC_DIR="${STDDFT_DIR}/casidapy"

# Prefer explicit PYTHON_EXE, then active venv, then python3 on PATH
if [ -z "${PYTHON_EXE:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON_EXE="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    fi
fi
if [ -z "${PYTHON_EXE}" ] || [ ! -x "${PYTHON_EXE}" ]; then
    echo "FATAL: No Python interpreter found. Activate your env or set PYTHON_EXE."
    exit 1
fi

NPROCS=${SLURM_NTASKS:-4}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "============================================================"
echo "  Casida TDDFT Calculation (MPI, CPU)"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-interactive}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Date:        $(date)"
echo "MPI ranks:   ${NPROCS}"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Python:      ${PYTHON_EXE}"
echo "Args:        $@"
echo "============================================================"
echo ""

cd "${STDDFT_DIR}"

if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Inside a Slurm allocation, srun uses scheduler-provided resources.
    srun --mpi=pmix -n ${NPROCS} ${PYTHON_EXE} -m casidapy.run_casida_parallel_generic "$@"
else
    # Interactive fallback outside Slurm.
    mpirun --oversubscribe -np ${NPROCS} ${PYTHON_EXE} -m casidapy.run_casida_parallel_generic "$@"
fi

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "FATAL: Casida calculation failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

echo ""
echo "============================================================"
echo "  Casida TDDFT complete at $(date)"
echo "============================================================"
