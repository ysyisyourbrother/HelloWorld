#!/usr/bin/env bash
# 将 Orin 边缘设备上产生的结果同步回本仓库（使用 ~/.ssh/config 中的 Host orin）
#
# 用法:
#   ./sync_from_orin.sh [选项]
#
# 选项:
#   -n, --dry-run          仅打印将要传输的内容，不实际同步
#   -s, --src PATH        远端基础目录（默认: ~/CodeSpace/Symphony）
#   -l, --local PATH      本地基础目录（默认: 当前仓库根目录）
#   -e, --exclude PATTERN  排除规则，可重复；传给 rsync 的 --exclude
#   -f, --exclude-from FILE  从文件读取排除规则（每行一条，# 开头为注释）
#   -h, --help             显示帮助
#
# 环境变量:
#   ORIN_HOST   SSH Host 名（默认: orin）
#   ORIN_SRC    远端基础目录（可被 --src 覆盖）
#   LOCAL_DEST  本地基础目录（可被 --local 覆盖）
#   ORIN_RESULT_DIRS 结果目录列表（空格分隔；默认: benchmark_results motivation_results motivation_results_symphony）
#
# 若存在 ./rsync_from_orin_exclude.txt，会自动作为 --exclude-from 追加（可在该文件中维护常用排除项）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ORIN_HOST="${ORIN_HOST:-orin}"
ORIN_SRC="${ORIN_SRC:-~/CodeSpace/Symphony}"
LOCAL_DEST="${LOCAL_DEST:-$PROJECT_ROOT}"

DRY_RUN=()
EXCLUDES=()
EXCLUDE_FILES=()

usage() {
  sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --dry-run)
      DRY_RUN=(--dry-run)
      shift
      ;;
    -s | --src)
      ORIN_SRC="${2:?需要 --src 参数}"
      shift 2
      ;;
    -l | --local)
      LOCAL_DEST="${2:?需要 --local 参数}"
      shift 2
      ;;
    -e | --exclude)
      EXCLUDES+=(--exclude "${2:?需要 --exclude 参数}")
      shift 2
      ;;
    -f | --exclude-from)
      EXCLUDE_FILES+=(--exclude-from "${2:?需要 --exclude-from 参数}")
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

DEFAULT_EXCLUDE_FILE="${PROJECT_ROOT}/rsync_from_orin_exclude.txt"
if [[ -f "$DEFAULT_EXCLUDE_FILE" ]]; then
  EXCLUDE_FILES+=(--exclude-from "$DEFAULT_EXCLUDE_FILE")
fi

RESULT_DIRS_DEFAULT=(
  benchmark_results
  motivation_results
  motivation_results_symphony
)

if [[ -n "${ORIN_RESULT_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  RESULT_DIRS=(${ORIN_RESULT_DIRS})
else
  RESULT_DIRS=("${RESULT_DIRS_DEFAULT[@]}")
fi

# 常见无需拉回本地的内容（在结果目录内也尽量避免同步缓存/进程文件）
BUILTIN_EXCLUDES=(
)

for d in "${RESULT_DIRS[@]}"; do
  mkdir -p "${LOCAL_DEST}/${d}"

  # 远端目录不存在时，rsync 会在进入源目录阶段直接失败（code 23），因此这里先判断并跳过
  if ! ssh -o BatchMode=yes "${ORIN_HOST}" "test -d ${ORIN_SRC}/${d}" >/dev/null 2>&1; then
    echo "跳过缺失目录: ${ORIN_HOST}:${ORIN_SRC}/${d}"
    continue
  fi

  echo "同步目录: ${d}"
  rsync -avz --delete \
    "${DRY_RUN[@]}" \
    "${BUILTIN_EXCLUDES[@]}" \
    "${EXCLUDES[@]}" \
    "${EXCLUDE_FILES[@]}" \
    --ignore-missing-args \
    -e ssh \
    "${ORIN_HOST}:${ORIN_SRC}/${d}/" \
    "${LOCAL_DEST}/${d}/"
done

echo "完成: ${ORIN_HOST}:${ORIN_SRC} -> ${LOCAL_DEST}"

