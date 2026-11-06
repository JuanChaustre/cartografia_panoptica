#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_URL="https://github.com/JuanChaustre/cartografia_panoptica/releases/download/modelos-v1.0.0"
ARCHIVE_NAME="cartografia_modelos_v1.tar.gz"
MASK2FORMER_DIR="$PROJECT_ROOT/training/runs/mask2former_stuff_reentrenado/best_model"
EFFICIENTPS_CHECKPOINT="$PROJECT_ROOT/training/runs/efficientps_original_desde_cero/best_validation_loss.pth"

if [[ -f "$MASK2FORMER_DIR/model.safetensors" && -f "$EFFICIENTPS_CHECKPOINT" ]]; then
  echo "Los checkpoints de ambos modelos ya están instalados."
  exit 0
fi

if [[ -e "$MASK2FORMER_DIR" || -e "$EFFICIENTPS_CHECKPOINT" ]]; then
  echo "Hay una instalación parcial o archivos locales de modelos; no los sobrescribiré." >&2
  echo "Revisa las rutas indicadas en $PROJECT_ROOT/streamlit_app.py antes de continuar." >&2
  exit 1
fi

for command_name in curl sha256sum tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Falta el comando requerido: $command_name" >&2
    exit 1
  fi
done

temporary_dir="$(mktemp -d)"
trap 'rm -rf "$temporary_dir"' EXIT

curl --fail --location --retry 3 \
  "$RELEASE_URL/$ARCHIVE_NAME" \
  --output "$temporary_dir/$ARCHIVE_NAME"
curl --fail --location --retry 3 \
  "$RELEASE_URL/$ARCHIVE_NAME.sha256" \
  --output "$temporary_dir/$ARCHIVE_NAME.sha256"

(
  cd "$temporary_dir"
  sha256sum --check "$ARCHIVE_NAME.sha256"
)

if tar --list --file "$temporary_dir/$ARCHIVE_NAME" \
  | grep --extended-regexp --quiet '(^/|(^|/)\.\.(/|$))'; then
  echo "El archivo de modelos contiene rutas no permitidas." >&2
  exit 1
fi

tar --extract --gzip --file "$temporary_dir/$ARCHIVE_NAME" \
  --directory "$PROJECT_ROOT" \
  --no-same-owner

if [[ ! -f "$MASK2FORMER_DIR/model.safetensors" \
  || ! -f "$MASK2FORMER_DIR/config.json" \
  || ! -f "$EFFICIENTPS_CHECKPOINT" ]]; then
  echo "La descarga terminó, pero faltan archivos requeridos para inferencia." >&2
  exit 1
fi

echo "Pesos de Mask2Former y EfficientPS original instalados."
