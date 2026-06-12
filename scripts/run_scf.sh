#!/bin/bash

#SBATCH --job-name=QEpy_SCF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=main
#SBATCH --time=12:00:00
#SBATCH --output=scf_%j.out
#SBATCH --error=scf_%j.err

#
# Step 1: Generate inputs for Casida TDDFT using QEpy (CPU, no GPU)
#
# IMPORTANT: QEpy's Driver must be launched as a SINGLE Python process.
# QE parallelises internally via OpenMP/MKL threads.
# Using mpirun with >1 rank causes "gather_complex_grid: f_in too small"
# when extracting wavefunctions, because the distributed FFT gather
# buffer is mis-sized (smooth grid != dense grid for USPP/high ecutrho).
#
# Usage:
#   sbatch scripts/run_scf.sh --config inputs.json
#   sbatch scripts/run_scf.sh --geometry examples/AG4_PARALLEL/ag4.vasp --pseudo examples/AG4_PARALLEL/ag_pbe_v1.4.uspp.F.UPF --workdir examples/AG4_PARALLEL
#   sbatch scripts/run_scf.sh --config inputs.json --ecutwfc 40 --nbnd 100
#
# Next step: run Casida with scripts/run_casida.sh or scripts/run_full_pipeline.sh using
# the same --workdir / output-prefix so rho_scf_* / psi_* / eig_* / occs_* match.
#
# Output files (in workdir):
#   rho_scf_<prefix>.xsf   — ground-state density
#   psi_<prefix>.npy        — KS wavefunctions
#   eig_<prefix>.npy        — KS eigenvalues (Hartree)
#   occs_<prefix>.npy       — occupation numbers
#   beta_projectors_<prefix>_<El>.npy  — USPP beta projectors (if ultrasoft)
#   qij_augmentation_<prefix>_<El>.npy — USPP Q_ij (if ultrasoft)
#

set -e

# ============================================================
# Environment
# ============================================================
if [ -d /projects/community/modulefiles ]; then
    module use /projects/community/modulefiles
fi
module load gcc/10.2.0/openmpi/4.0.5-bz186 2>/dev/null || true

if [ -f "${STDDFT_VENV:-/cache/home/am4655/edftpy_proj/edft}/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${STDDFT_VENV:-/cache/home/am4655/edftpy_proj/edft}/bin/activate"
fi
export PATH PYTHONPATH VIRTUAL_ENV

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    # In sbatch jobs, use the directory where user submitted the job.
    STDDFT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    STDDFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SRC_DIR="${STDDFT_DIR}/src"
if [ -z "${PYTHON_EXE:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON_EXE="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    fi
fi
if [ -z "${PYTHON_EXE}" ] || [ ! -x "${PYTHON_EXE}" ]; then
    echo "FATAL: No Python interpreter found. Set PYTHON_EXE or activate a venv."
    exit 1
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

echo "============================================================"
echo "  QEpy SCF Input Generation (CPU, serial Python + OMP)"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-interactive}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "Date:        $(date)"
echo "OMP threads: ${OMP_NUM_THREADS}"
echo "Python:      ${PYTHON_EXE}"
echo "Args:        $@"
echo "============================================================"
echo ""

cd "${STDDFT_DIR}"

# Run as a single process — QEpy/QE uses OpenMP threads internally.
# Do NOT use mpirun here; it causes gather_complex_grid errors on
# wavefunction extraction when ecutrho >> 4*ecutwfc (e.g. USPP).
${PYTHON_EXE} "${SRC_DIR}/generate_inputs_qepy.py" "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "FATAL: SCF failed with exit code ${EXIT_CODE}"
    exit $EXIT_CODE
fi

# ---- Check output files ----
# Try to figure out the output prefix and workdir from args
OUTPUT_PREFIX=""
WORKDIR="."
CONFIG_FILE=""

for i in $(seq 1 $#); do
    arg="${!i}"
    next_idx=$((i+1))
    case "$arg" in
        --output-prefix) [ $next_idx -le $# ] && OUTPUT_PREFIX="${!next_idx}" ;;
        --output-prefix=*) OUTPUT_PREFIX="${arg#*=}" ;;
        --workdir) [ $next_idx -le $# ] && WORKDIR="${!next_idx}" ;;
        --workdir=*) WORKDIR="${arg#*=}" ;;
        --geometry) [ $next_idx -le $# ] && GEOM="${!next_idx}" ;;
        --geometry=*) GEOM="${arg#*=}" ;;
        --config) [ $next_idx -le $# ] && CONFIG_FILE="${!next_idx}" ;;
        --config=*) CONFIG_FILE="${arg#*=}" ;;
    esac
done

# Read from config if needed
if [ -n "$CONFIG_FILE" ]; then
    CFG="$CONFIG_FILE"
    [ ! -f "$CFG" ] && CFG="${STDDFT_DIR}/${CONFIG_FILE}"
    if [ -f "$CFG" ]; then
        [ -z "$OUTPUT_PREFIX" ] && OUTPUT_PREFIX=$(${PYTHON_EXE} -c "import json; print(json.load(open('$CFG')).get('output-prefix',''))" 2>/dev/null)
        [ -z "$WORKDIR" ] || [ "$WORKDIR" = "." ] && WORKDIR=$(${PYTHON_EXE} -c "import json; print(json.load(open('$CFG')).get('workdir','.'))" 2>/dev/null)
        [ -z "$GEOM" ] && GEOM=$(${PYTHON_EXE} -c "import json; print(json.load(open('$CFG')).get('geometry',''))" 2>/dev/null)
    fi
fi

# Infer prefix from geometry if not set
[ -z "$OUTPUT_PREFIX" ] && [ -n "$GEOM" ] && OUTPUT_PREFIX=$(basename "$GEOM" | sed 's/\.[^.]*$//')

if [ -n "$OUTPUT_PREFIX" ]; then
    WORKDIR_ABS=$(cd "$WORKDIR" 2>/dev/null && pwd || echo "$WORKDIR")
    echo ""
    echo "Output files:"
    for f in rho_scf_${OUTPUT_PREFIX}.xsf psi_${OUTPUT_PREFIX}.npy eig_${OUTPUT_PREFIX}.npy occs_${OUTPUT_PREFIX}.npy; do
        FPATH="${WORKDIR_ABS}/${f}"
        if [ -f "${FPATH}" ]; then
            echo "  ✓ ${f}  ($(ls -lh ${FPATH} | awk '{print $5}'))"
        else
            echo "  ✗ ${f}  NOT FOUND"
        fi
    done
    # USPP files
    for f in ${WORKDIR_ABS}/beta_projectors_${OUTPUT_PREFIX}*.npy ${WORKDIR_ABS}/qij_augmentation_${OUTPUT_PREFIX}*.npy; do
        [ -f "$f" ] && echo "  ✓ $(basename $f)  ($(ls -lh $f | awk '{print $5}'))"
    done
fi

echo ""
echo "============================================================"
echo "  SCF complete at $(date)"
echo "============================================================"
