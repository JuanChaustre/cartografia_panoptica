#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="$PROJECT_ROOT/third_party/EfficientPS"
ENVIRONMENT_FILE="$PROJECT_ROOT/training/efficientps_original/environment.yml"
ENV_PREFIX="${EFFICIENTPS_ENV_PREFIX:-$HOME/.cache/cartografia-panoptica/efficientps-original}"
CUDA_HOME="$ENV_PREFIX/cuda"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-$HOME/.cache/micromamba}"
CUDA_INSTALLER_NAME="cuda_11.1.1_455.32.00_linux.run"
CUDA_INSTALLER_URL="https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/$CUDA_INSTALLER_NAME"
CUDA_INSTALLER_SIZE=3532172082
CUDA_INSTALLER_MD5="c24e2755e3868692051a38797ce01044"

if [[ ! -d "$UPSTREAM_ROOT/.git" ]]; then
  echo "No encuentro el repositorio oficial EfficientPS en $UPSTREAM_ROOT." >&2
  exit 1
fi

LOCAL_BIN="$HOME/.local/bin"
if [[ -x "$LOCAL_BIN/micromamba" ]]; then
  export PATH="$LOCAL_BIN:$PATH"
fi
if ! command -v micromamba >/dev/null 2>&1; then
  mkdir -p "$LOCAL_BIN"
  curl --fail --location --silent --show-error \
    https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$LOCAL_BIN" --strip-components=1 bin/micromamba
  export PATH="$LOCAL_BIN:$PATH"
fi

export MAMBA_ROOT_PREFIX="$MAMBA_ROOT"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  micromamba create --yes --prefix "$ENV_PREFIX" --file "$ENVIRONMENT_FILE"
else
  echo "Usando el entorno existente: $ENV_PREFIX"
  micromamba install --yes --prefix "$ENV_PREFIX" --file "$ENVIRONMENT_FILE"
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  CUDA_DOWNLOAD_DIR="$MAMBA_ROOT/downloads"
  CUDA_INSTALLER="$CUDA_DOWNLOAD_DIR/$CUDA_INSTALLER_NAME"
  mkdir -p "$CUDA_DOWNLOAD_DIR"
  if [[ ! -f "$CUDA_INSTALLER" ]] \
    || [[ "$(stat --format=%s "$CUDA_INSTALLER")" != "$CUDA_INSTALLER_SIZE" ]]; then
    echo "Descargando el toolkit oficial CUDA 11.1.1 (~3.53 GB)..."
    curl --fail --location --retry 3 --continue-at - \
      --output "$CUDA_INSTALLER" "$CUDA_INSTALLER_URL"
  else
    echo "Usando el instalador CUDA 10.2 ya descargado."
  fi

  actual_size="$(stat --format=%s "$CUDA_INSTALLER")"
  if [[ "$actual_size" != "$CUDA_INSTALLER_SIZE" ]]; then
    echo "Tamaño inesperado del instalador CUDA: $actual_size bytes." >&2
    exit 1
  fi
  actual_md5="$(md5sum "$CUDA_INSTALLER" | cut -d ' ' -f 1)"
  if [[ "$actual_md5" != "$CUDA_INSTALLER_MD5" ]]; then
    echo "La suma MD5 del instalador CUDA no coincide con el ETag oficial." >&2
    exit 1
  fi

  sh "$CUDA_INSTALLER" --nox11 --silent --override --toolkit \
    --toolkitpath="$CUDA_HOME" --librarypath="$CUDA_HOME/lib64"
  rm "$CUDA_INSTALLER"
fi

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "La instalación de CUDA no creó $CUDA_HOME/bin/nvcc." >&2
  exit 1
fi
"$CUDA_HOME/bin/nvcc" --version | grep -q 'release 11\.1' || {
  echo "Se requiere nvcc 11.1 para este entorno de PyTorch 1.9.0+cu111." >&2
  exit 1
}

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export FORCE_CUDA=1
export CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$ENV_PREFIX/lib:$ENV_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$ENV_PREFIX/include${CPATH:+:$CPATH}"
export LDFLAGS="-L/lib/x86_64-linux-gnu${LDFLAGS:+ $LDFLAGS}"

"$ENV_PREFIX/bin/python" -m pip install \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
"$ENV_PREFIX/bin/python" -m pip install \
  pycocotools cityscapesscripts==2.2.0 \
  git+https://github.com/mapillary/inplace_abn.git@24fc791e6d4796a1639e7a5dce6fa67377e51a3e

find "$UPSTREAM_ROOT/mmdet/ops" -type f -name '*.so' -delete
rm -rf -- "$UPSTREAM_ROOT/build"

(
  cd "$UPSTREAM_ROOT/efficientNet"
  "$ENV_PREFIX/bin/python" setup.py develop
)
(
  cd "$UPSTREAM_ROOT"
  "$ENV_PREFIX/bin/python" setup.py develop
)

"$ENV_PREFIX/bin/python" - <<'PY'
import torch
import geffnet
import mmdet
from mmdet.ops import nms

assert torch.cuda.is_available(), "CUDA no está disponible en este entorno."
probe = torch.ones(1, device="cuda")
assert float((probe + 1).item()) == 2.0
boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0, 0.9], [1.0, 1.0, 9.0, 9.0, 0.8]],
    device="cuda",
)
_, kept = nms(boxes, 0.5)
assert kept.numel() == 1, "La extensión CUDA NMS no pasó la prueba."
print("EfficientPS original:", mmdet.__version__)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("geffnet:", geffnet.__file__)
print("La extensión CUDA NMS ejecutó correctamente.")
PY

echo "Entorno instalado: $ENV_PREFIX"
