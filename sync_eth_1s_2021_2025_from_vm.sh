#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ubuntu@167.126.0.127}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/ubuntu/backtest-historico-bench}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/diego/backtest historico}"
SYMBOL="ETHUSDT"

LOCAL_BASE="${LOCAL_ROOT}/data/velas crudas/${SYMBOL}"
MANIFEST="${LOCAL_BASE}/manifest_eth_1s_2021_2025_local_sync.csv"

mkdir -p "${LOCAL_BASE}" "${LOCAL_BASE}/anual"

tmp_list="$(mktemp)"
trap 'rm -f "${tmp_list}"' EXIT

# Lista de archivos remotos objetivo (determinística): mensuales 2021..2025 y anuales 2021..2025.
ssh -o BatchMode=yes "${REMOTE_HOST}" "
  for y in 2021 2022 2023 2024 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
      p='${REMOTE_ROOT}/data/velas crudas/${SYMBOL}/${SYMBOL}_'\"\${y}\"\${m}'_1s_ohlc.parquet'
      [ -s \"\$p\" ] && echo \"\$p\"
    done
  done
  for y in 2021 2022 2023 2024 2025; do
    p='${REMOTE_ROOT}/data/velas crudas/${SYMBOL}/anual/${SYMBOL}_'\"\${y}\"'_1s_ohlc.parquet'
    [ -s \"\$p\" ] && echo \"\$p\"
  done
" | sort -u > "${tmp_list}"

echo "remote_path,local_path,action,remote_sha256,local_sha256,status,error_msg" > "${MANIFEST}"
echo "Remote targets=$(wc -l < "${tmp_list}")"

copied=0
skipped=0
errors=0

while IFS= read -r remote_file; do
  [[ -z "${remote_file}" ]] && continue

  rel="${remote_file#${REMOTE_ROOT}/}"
  local_file="${LOCAL_ROOT}/${rel}"
  mkdir -p "$(dirname "${local_file}")"

  remote_sha="$(ssh -n -o BatchMode=yes "${REMOTE_HOST}" "sha256sum '${remote_file}' 2>/dev/null | cut -d' ' -f1" || true)"
  action=""
  status="ok"
  err=""
  local_sha=""

  if [[ -s "${local_file}" ]]; then
    local_sha="$(sha256sum "${local_file}" | cut -d' ' -f1)"
    if [[ -n "${remote_sha}" && "${local_sha}" == "${remote_sha}" ]]; then
      action="skipped"
      skipped=$((skipped + 1))
    else
      action="error"
      status="error"
      err="checksum_mismatch_existing_local"
      errors=$((errors + 1))
    fi
  else
    tmp_local="${local_file}.part"
    rm -f "${tmp_local}"
    if ssh -n -o BatchMode=yes "${REMOTE_HOST}" "cat '${remote_file}'" > "${tmp_local}"; then
      mv "${tmp_local}" "${local_file}"
      local_sha="$(sha256sum "${local_file}" | cut -d' ' -f1)"
      if [[ -n "${remote_sha}" && "${local_sha}" == "${remote_sha}" ]]; then
        action="copied"
        copied=$((copied + 1))
      else
        action="error"
        status="error"
        err="checksum_mismatch_after_copy"
        errors=$((errors + 1))
      fi
    else
      rm -f "${tmp_local}"
      action="error"
      status="error"
      err="copy_failed"
      errors=$((errors + 1))
    fi
  fi

  echo "${remote_file},${local_file},${action},${remote_sha},${local_sha},${status},${err}" >> "${MANIFEST}"
done < "${tmp_list}"

echo "Manifest -> ${MANIFEST}"
echo "Copied=${copied} Skipped=${skipped} Errors=${errors}"

if [[ ${errors} -gt 0 ]]; then
  exit 1
fi
exit 0
