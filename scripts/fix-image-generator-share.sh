#!/usr/bin/env bash
set -euo pipefail

FSTAB_PATH="/etc/fstab"
BACKUP_PATH="/etc/fstab.pipeworks-image-generator-share.bak.$(date +%Y%m%d-%H%M%S)"

RUNTIME_ROOT="/srv/work/pipeworks/runtime/image-generator"
SHARE_ROOT="/srv/share/image-generator"

GALLERY_SRC="${RUNTIME_ROOT}/gallery"
LORA_RUNS_SRC="${RUNTIME_ROOT}/outputs/lora_runs"
GALLERY_DB_SRC="${RUNTIME_ROOT}/gallery.json"

GALLERY_DST="${SHARE_ROOT}/gallery"
LORA_RUNS_DST="${SHARE_ROOT}/lora_runs"
GALLERY_DB_DST="${SHARE_ROOT}/gallery.json"

OLD_BAD_LINE="${RUNTIME_ROOT}/outputs ${GALLERY_DST} none bind,ro 0 0"
NEW_GALLERY_LINE="${GALLERY_SRC} ${GALLERY_DST} none bind,ro 0 0"
NEW_LORA_LINE="${LORA_RUNS_SRC} ${LORA_RUNS_DST} none bind,ro 0 0"
NEW_DB_LINE="${GALLERY_DB_SRC} ${GALLERY_DB_DST} none bind,ro 0 0"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run this script with sudo."
    exit 1
  fi
}

require_paths() {
  local path
  for path in "${GALLERY_SRC}" "${LORA_RUNS_SRC}" "${GALLERY_DB_SRC}"; do
    if [[ ! -e "${path}" ]]; then
      echo "Required source path is missing: ${path}"
      exit 1
    fi
  done
}

ensure_share_root() {
  mkdir -p "${SHARE_ROOT}"
}

backup_fstab() {
  cp -a "${FSTAB_PATH}" "${BACKUP_PATH}"
  echo "Backed up ${FSTAB_PATH} to ${BACKUP_PATH}"
}

rewrite_fstab() {
  python3 - "${FSTAB_PATH}" "${OLD_BAD_LINE}" "${NEW_GALLERY_LINE}" "${NEW_LORA_LINE}" "${NEW_DB_LINE}" <<'PY'
from pathlib import Path
import sys

fstab_path = Path(sys.argv[1])
old_bad_line = sys.argv[2]
new_gallery_line = sys.argv[3]
new_lora_line = sys.argv[4]
new_db_line = sys.argv[5]


def normalize_fields(line: str) -> str:
    return " ".join(line.split())

lines = fstab_path.read_text(encoding="utf-8").splitlines()
filtered: list[str] = []

entries_to_remove = {
    normalize_fields(old_bad_line),
    normalize_fields(new_gallery_line),
    normalize_fields(new_lora_line),
    normalize_fields(new_db_line),
}

for line in lines:
    normalized = normalize_fields(line)
    if normalized in entries_to_remove:
        continue
    filtered.append(line)

if filtered and filtered[-1] != "":
    filtered.append("")

filtered.extend([new_gallery_line, new_lora_line, new_db_line, ""])
fstab_path.write_text("\n".join(filtered), encoding="utf-8")
PY
}

ensure_mountpoints() {
  mkdir -p "${GALLERY_DST}" "${LORA_RUNS_DST}"
  if [[ ! -e "${GALLERY_DB_DST}" ]]; then
    touch "${GALLERY_DB_DST}"
  fi
}

unmount_old_paths() {
  local target
  for target in "${LORA_RUNS_DST}" "${GALLERY_DB_DST}" "${GALLERY_DST}"; do
    if findmnt -rn "${target}" >/dev/null 2>&1; then
      if umount "${target}"; then
        echo "Unmounted ${target}"
      else
        echo "Unmount failed for ${target}; retrying with lazy unmount"
        umount -l "${target}"
        echo "Lazy-unmounted ${target}"
      fi
    fi
  done
}

remount_all() {
  mount --bind "${GALLERY_SRC}" "${GALLERY_DST}"
  mount -o remount,bind,ro "${GALLERY_DST}"
  mount --bind "${LORA_RUNS_SRC}" "${LORA_RUNS_DST}"
  mount -o remount,bind,ro "${LORA_RUNS_DST}"
  mount --bind "${GALLERY_DB_SRC}" "${GALLERY_DB_DST}"
  mount -o remount,bind,ro "${GALLERY_DB_DST}"
}

show_result() {
  echo
  echo "Current mounts under ${SHARE_ROOT}:"
  findmnt -R "${SHARE_ROOT}" || true
  echo
  echo "Shared tree:"
  find "${SHARE_ROOT}" -maxdepth 2 -printf '%P|%y\n' | sort
}

main() {
  require_root
  require_paths
  ensure_share_root
  backup_fstab
  rewrite_fstab
  unmount_old_paths
  ensure_mountpoints
  remount_all
  show_result
}

main "$@"
