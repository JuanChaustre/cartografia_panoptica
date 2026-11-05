#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="$PROJECT_ROOT/third_party/EfficientPS"
ENV_PREFIX="${EFFICIENTPS_ENV_PREFIX:-$HOME/.cache/cartografia-panoptica/efficientps-original}"
METRICS_PYTHON="${METRICS_PYTHON:-$PROJECT_ROOT/.venv-mask2former/bin/python}"
RUN_ROOT="${WORK_DIR:-$PROJECT_ROOT/training/runs/efficientps_original}"
RESUME_FROM="${RESUME_FROM:-}"
CONFIG="$PROJECT_ROOT/training/efficientps_original/efficientps_cartografia.py"
RAW_TEST_PREDICTIONS="$RUN_ROOT/test_raw_predictions"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "Falta el entorno; ejecuta tools/instalar_efficientps_original.sh primero." >&2
  exit 1
fi
if [[ ! -x "$METRICS_PYTHON" ]]; then
  echo "No encuentro el entorno Python moderno de métricas: $METRICS_PYTHON" >&2
  echo "Define METRICS_PYTHON con un Python 3.10+ que tenga las dependencias del proyecto." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_ROOT/data_panoptic_stuff/splits.json" ]]; then
  echo "Falta data_panoptic_stuff/splits.json." >&2
  exit 1
fi
if [[ -n "$RESUME_FROM" ]]; then
  if [[ ! -f "$RESUME_FROM" ]]; then
    echo "No existe el checkpoint indicado para reanudar: $RESUME_FROM" >&2
    exit 1
  fi
  mkdir -p "$RUN_ROOT"
  RUN_ROOT_REAL="$(realpath -m "$RUN_ROOT")"
  CHECKPOINT_REAL="$(realpath "$RESUME_FROM")"
  if [[ "$(dirname "$CHECKPOINT_REAL")" != "$RUN_ROOT_REAL" ]]; then
    echo "El checkpoint de reanudación debe estar dentro de WORK_DIR." >&2
    exit 1
  fi
  if [[ ! -f "$RUN_ROOT/loss_history.csv" ]]; then
    echo "No encuentro loss_history.csv; no puedo reanudar early stopping." >&2
    exit 1
  fi
elif [[ -f "$RUN_ROOT/latest.pth" ]]; then
  echo "Ya existe $RUN_ROOT/latest.pth; para continuar define RESUME_FROM o elige otro WORK_DIR." >&2
  exit 1
fi

"$METRICS_PYTHON" "$PROJECT_ROOT/tools/preparar_efficientps_original.py" \
  --overwrite
mkdir -p "$RUN_ROOT"

BEST_CHECKPOINT="$RUN_ROOT/best_validation_loss.pth"
if [[ -n "$RESUME_FROM" && ! -f "$RUN_ROOT/best_validation_loss.pth" ]]; then
  "$METRICS_PYTHON" - "$RUN_ROOT/loss_history.csv" "$RUN_ROOT" "$BEST_CHECKPOINT" <<'PY'
import csv
import shutil
import sys
from pathlib import Path

history_path, work_dir, destination = map(Path, sys.argv[1:])
with history_path.open(newline="", encoding="utf-8") as file:
    history = list(csv.DictReader(file))
if not history:
    raise SystemExit("No hay historial para restaurar el mejor checkpoint.")
best = min(history, key=lambda row: float(row["validation_loss"]))
source = work_dir / "epoch_{}.pth".format(int(float(best["epoch"])))
if not source.is_file():
    raise SystemExit("No existe el checkpoint de la mejor época: {}".format(source))
shutil.copy2(source, work_dir / "best_validation_loss.pth")
print("Checkpoint histórico de menor pérdida restaurado.")
PY
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export FORCE_CUDA=1
export PYTHONPATH="$UPSTREAM_ROOT:$PROJECT_ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
export EFFICIENTPS_PANOPTIC_OUTPUT_DIR="$RUN_ROOT/validation_raw_predictions"
export CARTOGRAFIA_PROJECT_ROOT="$PROJECT_ROOT"

TRAIN_ARGS=(tools/train.py "$CONFIG" --work_dir "$RUN_ROOT" --validate --seed 42)
if [[ -n "$RESUME_FROM" ]]; then
  TRAIN_ARGS+=(--resume_from "$RESUME_FROM")
fi
(
  cd "$UPSTREAM_ROOT"
  "$ENV_PREFIX/bin/python" "${TRAIN_ARGS[@]}" \
    2>&1 | tee -a "$RUN_ROOT/training.log"
)

if [[ ! -f "$RUN_ROOT/latest.pth" ]]; then
  echo "El entrenamiento terminó sin crear latest.pth." >&2
  exit 1
fi

if [[ ! -f "$BEST_CHECKPOINT" ]]; then
  echo "No se creó el checkpoint con menor pérdida de validación." >&2
  exit 1
fi

"$ENV_PREFIX/bin/python" "$PROJECT_ROOT/tools/evaluar_efficientps_original.py" \
  --config "$CONFIG" \
  --checkpoint "$BEST_CHECKPOINT" \
  --split test \
  --output-root "$RAW_TEST_PREDICTIONS" \
  --score-threshold 0.5 \
  2>&1 | tee "$RUN_ROOT/test.log"

"$METRICS_PYTHON" \
  "$PROJECT_ROOT/training/efficientps_original/export_metrics.py" \
  --prediction-dir "$RAW_TEST_PREDICTIONS" \
  --output-dir "$RUN_ROOT/test_predictions"

echo "Entrenamiento y evaluación completados: $RUN_ROOT"
