#!/bin/bash

#SBATCH --job-name=STDDFT_Pipeline
#SBATCH --nodes=1
# One MPI rank per task for Casida (--ntasks becomes mpirun -np). OpenMP during
# QEpy SCF uses all CPUs on the node via SLURM_CPUS_ON_NODE in the script below.
# Tune for your cluster (--ntasks-per-node / max memory per node).
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --partition=main
#SBATCH --time=02:00:00
#SBATCH --output=pipeline_%j.out
#SBATCH --error=pipeline_%j.err

#
# Full STDDFT pipeline: QEpy SCF (step 1) -> Casida TDDFT (step 2, MPI, CPU only).
# Casida uses the same active-space rules as casida_engine: --n-total-occ sets the LUMO
# index (HOMO count); omit it to take the lowest --n-occ occupied bands from the file.
#
# Usage:
#   sbatch scripts/run_full_pipeline.sh \
#       --geometry examples/AG4_PARALLEL/ag4.vasp \
#       --pseudo "Ag:Ag_ONCV_PBE-1.2.upf" \
#       --workdir examples/AG4_PARALLEL \
#       --output-prefix ag4 \
#       --nbnd 70 \
#       --nc-pseudo "Ag:Ag_ONCV_PBE-1.2.upf" \
#       --n-occ 10 --n-unocc 20 --n-states 30 \
#       --n-total-occ 38
#
#   # JSON config (same keys as generate_inputs_qepy.py):
#   sbatch scripts/run_full_pipeline.sh --config my_config.json --nc-pseudo "Ag:Ag_ONCV.upf"
#
# Casida requires norm-conserving pseudopotentials via --pseudo-map. Pass either:
#   --nc-pseudo "H:H_ONCV.upf,O:O_ONCV.upf"
# or rely on --pseudo when it is already in Elem:file[,Elem:file] form, or a single NC file
# (element inferred from the basename).
#
# Arguments:
#   [SCF args]     -> passed to generate_inputs_qepy.py
#   [Casida flags] -> --n-occ, --n-unocc, --n-states, --n-total-occ, --xc, --matrix-free, ...
#                     see case statement below
#

set -e

# ============================================================
# Environment setup
# ============================================================
if [ -d /projects/community-old/modulefiles ]; then
    module use /projects/community-old/modulefiles
fi
module load gcc/10.2.0/openmpi/4.0.5-bz186 2>/dev/null || true

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
SRC_DIR="${STDDFT_DIR}/src"
if [ -z "${PYTHON_EXE:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYTHON_EXE="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON_EXE="$(command -v python3 2>/dev/null || command -v python 2>/dev/null || true)"
    fi
fi
if [ -z "${PYTHON_EXE}" ] || [ ! -x "${PYTHON_EXE}" ]; then
    echo "FATAL: No Python interpreter found. Set PYTHON_EXE or STDDFT_VENV."
    exit 1
fi

echo "============================================================"
echo "  STDDFT Full Pipeline (MPI + CPU)"
echo "  Step 1: Generate inputs (QEpy SCF)"
echo "  Step 2: Run Casida TDDFT"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Date: $(date)"
echo "MPI ranks: ${SLURM_NTASKS:-4}"
echo "CPUs/task: ${SLURM_CPUS_PER_TASK:-8}"
echo "Python: ${PYTHON_EXE}"
echo ""

# ============================================================
# Parse arguments — separate SCF args from Casida-only args
# ============================================================
SCF_ARGS=()
CASIDA_N_OCC=""
CASIDA_N_UNOCC=""
CASIDA_N_TOTAL_OCC=""
CASIDA_N_STATES="50"
CASIDA_XC="PBE"
CASIDA_MATRIX_FREE=""
CASIDA_SOLVER_METHOD="lobpcg"
CASIDA_TDA=""
CASIDA_NC_PSEUDO=""
SKIP_CASIDA=""
SKIP_SCF=""
CASIDA_PLOT="--plot"
CASIDA_SIGMA="0.1"

WORKDIR=""
CONFIG_FILE=""
GEOMETRY=""
PSEUDO=""
OUTPUT_PREFIX=""
NBND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-occ)         CASIDA_N_OCC="$2"; shift 2 ;;
        --n-unocc)       CASIDA_N_UNOCC="$2"; shift 2 ;;
        --n-total-occ)   CASIDA_N_TOTAL_OCC="$2"; shift 2 ;;
        --n-states)      CASIDA_N_STATES="$2"; shift 2 ;;
        --xc)            CASIDA_XC="$2"; shift 2 ;;
        --matrix-free)   CASIDA_MATRIX_FREE="--matrix-free"; shift ;;
        --solver-method) CASIDA_SOLVER_METHOD="$2"; shift 2 ;;
        --tda)           CASIDA_TDA="--tda"; shift ;;
        --nc-pseudo)     CASIDA_NC_PSEUDO="$2"; shift 2 ;;
        --skip-casida)   SKIP_CASIDA="1"; shift ;;
        --skip-scf)      SKIP_SCF="1"; shift ;;
        --no-plot)       CASIDA_PLOT=""; shift ;;
        --sigma)         CASIDA_SIGMA="$2"; shift 2 ;;
        --workdir)       WORKDIR="$2";       SCF_ARGS+=("$1" "$2"); shift 2 ;;
        --config)        CONFIG_FILE="$2";   SCF_ARGS+=("$1" "$2"); shift 2 ;;
        --geometry)      GEOMETRY="$2";      SCF_ARGS+=("$1" "$2"); shift 2 ;;
        --pseudo)        PSEUDO="$2";        SCF_ARGS+=("$1" "$2"); shift 2 ;;
        --output-prefix) OUTPUT_PREFIX="$2"; SCF_ARGS+=("$1" "$2"); shift 2 ;;
        --nbnd)          NBND="$2";          SCF_ARGS+=("$1" "$2"); shift 2 ;;
        *)               SCF_ARGS+=("$1"); shift ;;
    esac
done

# ============================================================
# If config file provided, extract key values we need
# ============================================================
if [ -n "$CONFIG_FILE" ]; then
    CONFIG_PATH="$CONFIG_FILE"
    if [ ! -f "$CONFIG_PATH" ]; then
        CONFIG_PATH="${STDDFT_DIR}/${CONFIG_FILE}"
    fi
    if [ -f "$CONFIG_PATH" ]; then
        echo "Reading config: ${CONFIG_PATH}"
        [ -z "$WORKDIR" ]       && WORKDIR=$(${PYTHON_EXE} -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('workdir','.'))" 2>/dev/null)
        [ -z "$GEOMETRY" ]      && GEOMETRY=$(${PYTHON_EXE} -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('geometry',''))" 2>/dev/null)
        [ -z "$PSEUDO" ]        && PSEUDO=$(${PYTHON_EXE} -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('pseudo',''))" 2>/dev/null)
        [ -z "$OUTPUT_PREFIX" ] && OUTPUT_PREFIX=$(${PYTHON_EXE} -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('output-prefix',''))" 2>/dev/null)
        [ -z "$NBND" ]          && NBND=$(${PYTHON_EXE} -c "import json; d=json.load(open('$CONFIG_PATH')); print(d.get('nbnd',''))" 2>/dev/null)
    else
        echo "WARNING: Config file not found: ${CONFIG_FILE}"
    fi
fi

# Defaults
[ -z "$WORKDIR" ] && WORKDIR="."
[ -z "$OUTPUT_PREFIX" ] && [ -n "$GEOMETRY" ] && OUTPUT_PREFIX=$(basename "$GEOMETRY" | sed 's/\.[^.]*$//')

# Resolve workdir to absolute path
cd "${STDDFT_DIR}"
WORKDIR_ABS=$(cd "$WORKDIR" 2>/dev/null && pwd || echo "$WORKDIR")

echo ""
echo "Configuration:"
echo "  Workdir:       ${WORKDIR_ABS}"
echo "  Geometry:      ${GEOMETRY}"
echo "  Pseudo (SCF):  ${PSEUDO}"
echo "  Output prefix: ${OUTPUT_PREFIX}"
echo "  NBND:          ${NBND:-auto}"
echo "  Casida XC:     ${CASIDA_XC}"
echo "  NC map:        ${CASIDA_NC_PSEUDO:-(derive from --pseudo)}"
echo "  Matrix-free:   ${CASIDA_MATRIX_FREE:-dense}"
echo "  n_total_occ:   ${CASIDA_N_TOTAL_OCC:-auto from occs}"
echo ""

# ============================================================
# STEP 1: Generate inputs (SCF with QEpy)
# ============================================================
if [ -z "$SKIP_SCF" ]; then
    echo "============================================================"
    echo "  STEP 1: Generating inputs (QEpy SCF)"
    echo "============================================================"
    echo ""

    cd "${STDDFT_DIR}"

    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
    echo "Running QEpy SCF with ${OMP_NUM_THREADS} OpenMP threads"
    echo "SCF args: ${SCF_ARGS[*]}"
    echo ""

    ${PYTHON_EXE} "${SRC_DIR}/generate_inputs_qepy.py" "${SCF_ARGS[@]}"
    SCF_EXIT=$?

    if [ $SCF_EXIT -ne 0 ]; then
        echo ""
        echo "FATAL: SCF step failed with exit code ${SCF_EXIT}"
        exit $SCF_EXIT
    fi

    echo ""
    echo "SCF step completed successfully."
    echo ""

    echo "Checking SCF output files:"
    for f in rho_scf_${OUTPUT_PREFIX}.xsf psi_${OUTPUT_PREFIX}.npy eig_${OUTPUT_PREFIX}.npy occs_${OUTPUT_PREFIX}.npy; do
        FPATH="${WORKDIR_ABS}/${f}"
        if [ -f "${FPATH}" ]; then
            echo "  ✓ ${f} ($(ls -lh "${FPATH}" | awk '{print $5}'))"
        else
            echo "  ✗ ${f} NOT FOUND"
            echo "FATAL: Required output file missing. Cannot proceed to Casida."
            exit 1
        fi
    done
    echo ""
else
    echo "STEP 1: Skipped (--skip-scf)"
    echo ""
    for f in rho_scf_${OUTPUT_PREFIX}.xsf psi_${OUTPUT_PREFIX}.npy eig_${OUTPUT_PREFIX}.npy occs_${OUTPUT_PREFIX}.npy; do
        FPATH="${WORKDIR_ABS}/${f}"
        if [ ! -f "${FPATH}" ]; then
            echo "FATAL: Required file not found: ${FPATH}"
            echo "Run without --skip-scf to generate inputs first."
            exit 1
        fi
    done
fi

# ============================================================
# STEP 2: Run Casida TDDFT
# ============================================================
if [ -n "$SKIP_CASIDA" ]; then
    echo "STEP 2: Skipped (--skip-casida)"
    echo ""
    echo "Pipeline finished (SCF only)."
    exit 0
fi

echo "============================================================"
echo "  STEP 2: Running Casida TDDFT"
echo "============================================================"
echo ""

if [ -z "$CASIDA_N_OCC" ] || [ -z "$CASIDA_N_UNOCC" ]; then
    AUTODET=$(${PYTHON_EXE} -c "
import numpy as np
occs = np.load('${WORKDIR_ABS}/occs_${OUTPUT_PREFIX}.npy')
n_total_occ = int(np.sum(occs > 0.5))
n_total = len(occs)
n_unocc_avail = n_total - n_total_occ
n_occ = min(n_total_occ, ${CASIDA_N_OCC:-0} if ${CASIDA_N_OCC:-0} > 0 else n_total_occ)
n_unocc = min(n_unocc_avail, ${CASIDA_N_UNOCC:-0} if ${CASIDA_N_UNOCC:-0} > 0 else n_unocc_avail)
print(f'{n_occ} {n_unocc} {n_total_occ}')
" 2>/dev/null)
    AUTO_N_OCC=$(echo "$AUTODET" | awk '{print $1}')
    AUTO_N_UNOCC=$(echo "$AUTODET" | awk '{print $2}')
    AUTO_N_TOTAL_OCC=$(echo "$AUTODET" | awk '{print $3}')

    [ -z "$CASIDA_N_OCC" ]   && CASIDA_N_OCC=$AUTO_N_OCC
    [ -z "$CASIDA_N_UNOCC" ] && CASIDA_N_UNOCC=$AUTO_N_UNOCC

    echo "Auto-detected orbital counts:"
    echo "  Total occupied:     ${AUTO_N_TOTAL_OCC}"
    echo "  n_occ (Casida):     ${CASIDA_N_OCC}"
    echo "  n_unocc (Casida):   ${CASIDA_N_UNOCC}"
    echo "  Transitions:        $((CASIDA_N_OCC * CASIDA_N_UNOCC))"
    echo ""
fi

ATOMS_FILE=""
for ext in vasp xyz cif; do
    CANDIDATE="${WORKDIR_ABS}/$(basename "${GEOMETRY%.???}").${ext}"
    if [ -f "$CANDIDATE" ]; then
        ATOMS_FILE=$(basename "$CANDIDATE")
        break
    fi
done
[ -z "$ATOMS_FILE" ] && ATOMS_FILE=$(basename "$GEOMETRY")

echo "  Atoms file:       ${ATOMS_FILE}"
echo "  Density file:     rho_scf_${OUTPUT_PREFIX}.xsf"
echo "  Psi file:         psi_${OUTPUT_PREFIX}.npy"
echo ""

# ---- Norm-conserving --pseudo-map for Casida (required by run_casida_parallel_generic.py) ----
PSEUDO_MAP=""
if [ -n "$CASIDA_NC_PSEUDO" ]; then
    PSEUDO_MAP="$CASIDA_NC_PSEUDO"
elif [ -n "$PSEUDO" ]; then
    if [[ "$PSEUDO" == *","* ]]; then
        PSEUDO_MAP="$PSEUDO"
    elif [[ "$PSEUDO" == *":"* ]]; then
        PSEUDO_MAP="$PSEUDO"
    else
        PP_BASENAME=$(basename "$PSEUDO")
        PP_ELEM=$(echo "$PP_BASENAME" | sed 's/_.*//;s/\..*//' | awk '{print toupper(substr($0,1,1)) tolower(substr($0,2))}')
        PSEUDO_MAP="${PP_ELEM}:${PP_BASENAME}"
    fi
else
    echo "  FATAL: Casida needs --pseudo-map. Set --nc-pseudo \"El:file,...\" or --pseudo for SCF"
    echo "         (single NC file or Elem:file[,Elem:file] form)."
    exit 1
fi

echo "  Pseudo map (NC):  ${PSEUDO_MAP}"
echo ""

NPROCS=${SLURM_NTASKS:-4}

echo "Running Casida with ${NPROCS} MPI ranks (CPU)..."
echo ""

PSEUDO_MAP_ARG="--pseudo-map ${PSEUDO_MAP}"

CASIDA_TOTAL_ARGS=()
if [ -n "${CASIDA_N_TOTAL_OCC}" ]; then
    CASIDA_TOTAL_ARGS+=(--n-total-occ "${CASIDA_N_TOTAL_OCC}")
fi

mpirun -np "${NPROCS}" ${PYTHON_EXE} "${SRC_DIR}/run_casida_parallel_generic.py" \
    --workdir "${WORKDIR_ABS}" \
    --atoms "${ATOMS_FILE}" \
    --density "rho_scf_${OUTPUT_PREFIX}.xsf" \
    --psi "psi_${OUTPUT_PREFIX}.npy" \
    --eigs "eig_${OUTPUT_PREFIX}.npy" \
    --occs "occs_${OUTPUT_PREFIX}.npy" \
    ${PSEUDO_MAP_ARG} \
    --n-occ "${CASIDA_N_OCC}" \
    --n-unocc "${CASIDA_N_UNOCC}" \
    "${CASIDA_TOTAL_ARGS[@]}" \
    --n-states "${CASIDA_N_STATES}" \
    --xc "${CASIDA_XC}" \
    --output-prefix "casida_${OUTPUT_PREFIX}" \
    --sigma "${CASIDA_SIGMA}" \
    --solver-method "${CASIDA_SOLVER_METHOD}" \
    ${CASIDA_MATRIX_FREE} \
    ${CASIDA_TDA} \
    ${CASIDA_PLOT}

CASIDA_EXIT=$?

echo ""
if [ $CASIDA_EXIT -ne 0 ]; then
    echo "Casida step failed with exit code ${CASIDA_EXIT}"
    exit $CASIDA_EXIT
fi

echo ""
echo "============================================================"
echo "  Pipeline complete at $(date)"
echo "============================================================"
echo ""
echo "Output files in ${WORKDIR_ABS}:"
for f in rho_scf_${OUTPUT_PREFIX}.xsf \
         psi_${OUTPUT_PREFIX}.npy \
         eig_${OUTPUT_PREFIX}.npy \
         occs_${OUTPUT_PREFIX}.npy \
         casida_${OUTPUT_PREFIX}_omega.npy \
         casida_${OUTPUT_PREFIX}_f.npy \
         casida_${OUTPUT_PREFIX}_results.txt \
         casida_${OUTPUT_PREFIX}_spectrum.png; do
    FPATH="${WORKDIR_ABS}/${f}"
    if [ -f "${FPATH}" ]; then
        echo "  ✓ ${f} ($(ls -lh "${FPATH}" | awk '{print $5}'))"
    fi
done
echo ""
echo "Done!"
