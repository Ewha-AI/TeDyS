#!/usr/bin/env bash
set -euo pipefail

# ===================== OASIS paths =====================
DATA_CSV="./oasis2/oasis2_progression.csv"
CT_TEMPLATE="./oasis2/data/{patient_id}_{st}.npz"
FEATURE_ROOT="./oasis2/feat"

GPU=6
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

# ===================== training knobs =====================
EPOCHS=50
BS=8
BSEVAL=1
NW=8

LR="5e-6"
WD_LIST=(1e-2)
MS_SUFFIX="_nnunet_ms.npz"
MS_SCALES="res5"
MS_POOL="avg"
NUM_CLASSES=2
D_MODEL=256
DEPTH=2
HEADS=4
MLP_RATIO=4.0
DROPOUT=0.0
KQ=6
STD_OT_EPS=0.1
STD_OT_ITERS=20
LTD_ODE_WIDTH=256
LTD_ODE_STEPS=4
SLOT_ON=1
SLOT_HEADS=4
SLOT_ITERS=3
SLOT_NUM_LIST=(3)
TA_HEADS=4
TA_POOL_TOKENS=1
USE_AR_QUERY=0
AR_DT_MODE="add"
OUTROOT_BASE="./results/oasis2"
mkdir -p "${OUTROOT_BASE}"

find_resume_ckpt () {
  local d="$1"
  if [[ -f "${d}/checkpoint_last.pth" ]]; then echo "${d}/checkpoint_last.pth"; return; fi
  if [[ -f "${d}/ckpt_last.pth" ]]; then echo "${d}/ckpt_last.pth"; return; fi
  if [[ -f "${d}/last.pth" ]]; then echo "${d}/last.pth"; return; fi
  echo ""
}

FOLDS=(0 1 2 3 4)

echo "[sweep] runs = $((${#FOLDS[@]} * ${#WD_LIST[@]} * ${#SLOT_NUM_LIST[@]}))"
echo "[sweep] fixed LR: ${LR}"
echo "[sweep] WD_LIST: ${WD_LIST[*]}"
echo "============================================================"

for FOLD in "${FOLDS[@]}"; do
  for SLOT_NUM in "${SLOT_NUM_LIST[@]}"; do
    for WD in "${WD_LIST[@]}"; do

      RUN_NAME="oasis__fold${FOLD}__slot${SLOT_ON}__snum${SLOT_NUM}__siters${SLOT_ITERS}__STDot__LTDode__lr${LR}__wd${WD}__ar${USE_AR_QUERY}"
      OUTROOT="${OUTROOT_BASE}/${RUN_NAME}"
      mkdir -p "${OUTROOT}"

      RESUME_CKPT="$(find_resume_ckpt "${OUTROOT}")"
      RESUME_ARGS=()
      if [[ -n "${RESUME_CKPT}" ]]; then
        echo "[resume] found: ${RESUME_CKPT}"
        RESUME_ARGS+=( --resume "${RESUME_CKPT}" )
      fi

      echo "============================================================"
      echo "[GPU ${GPU}] OASIS | fold=${FOLD} | LR=${LR} WD=${WD} | slot=${SLOT_ON} snum=${SLOT_NUM} siters=${SLOT_ITERS} | STD=OT LTD=ODE | ar=${USE_AR_QUERY}"
      echo "OUTDIR: ${OUTROOT}"
      echo "============================================================"

      AR_ARGS=()
      if [[ "${USE_AR_QUERY}" == "1" ]]; then
        AR_ARGS+=( --qall_use_ar_query --qall_ar_dt_mode "${AR_DT_MODE}" )
      fi

      SLOT_ARGS=()
      if [[ "${SLOT_ON}" == "1" ]]; then
        SLOT_ARGS+=( --qall_use_slot --qall_slot_num "${SLOT_NUM}" --qall_slot_iters "${SLOT_ITERS}" --qall_slot_heads "${SLOT_HEADS}" )
      fi

      python main.py \
        --data_csv "${DATA_CSV}" \
        --dataset oasis \
        --ct_path_template "${CT_TEMPLATE}" \
        --use_multiscale_feature \
        --feature_root "${FEATURE_ROOT}" \
        --ms_suffix "${MS_SUFFIX}" \
        --ms_scales "${MS_SCALES}" \
        --ms_pool "${MS_POOL}" \
        --fold "${FOLD}" \
        --label_col prog_event \
        --outdir "${OUTROOT}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BS}" \
        --batch_size_eval "${BSEVAL}" \
        --lr "${LR}" \
        --weight_decay "${WD}" \
        --num_workers "${NW}" \
        --amp \
        --num_classes "${NUM_CLASSES}" \
        --qall_d_model "${D_MODEL}" \
        --qall_depth "${DEPTH}" \
        --qall_heads "${HEADS}" \
        --qall_mlp_ratio "${MLP_RATIO}" \
        --qall_dropout "${DROPOUT}" \
        --qall_num_query_tokens "${KQ}" \
        --qall_ta_heads "${TA_HEADS}" \
        --qall_ta_pool_tokens "${TA_POOL_TOKENS}" \
        --std_ot_eps "${STD_OT_EPS}" \
        --std_ot_iters "${STD_OT_ITERS}" \
        --ltd_ode_width "${LTD_ODE_WIDTH}" \
        --ltd_ode_steps "${LTD_ODE_STEPS}" \
        "${AR_ARGS[@]}" \
        "${SLOT_ARGS[@]}" \
        "${RESUME_ARGS[@]}"

    done
  done
done

echo "All done. Base results in: ${OUTROOT_BASE}"
