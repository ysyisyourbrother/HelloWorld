#!/usr/bin/env bash
# 将本仓库同步到 Orin 边缘设备（使用 ~/.ssh/config 中的 Host orin）
#
# 用法:
#   ./sync_to_orin.sh [选项]
#
# 选项:
#   -n, --dry-run          仅打印将要传输的内容，不实际同步
#   -d, --dest PATH        远端目录（默认: ~/HelloWorld）
#   -e, --exclude PATTERN  排除规则，可重复；传给 rsync 的 --exclude
#   -f, --exclude-from FILE  从文件读取排除规则（每行一条，# 开头为注释）
#   -h, --help             显示帮助
#
# 环境变量:
#   ORIN_HOST   SSH Host 名（默认: orin）
#   ORIN_DEST   远端路径（可被 --dest 覆盖）
#
# 若存在 ./rsync_orin_exclude.txt，会自动作为 --exclude-from 追加（可在该文件中维护常用排除项）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ORIN_HOST="${ORIN_HOST:-orin}"
ORIN_DEST="${ORIN_DEST:-~/CodeSpace/Symphony}"
DRY_RUN=()
EXCLUDES=()
EXCLUDE_FILES=()

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n | --dry-run)
      DRY_RUN=(--dry-run)
      shift
      ;;
    -d | --dest)
      ORIN_DEST="${2:?需要 --dest 参数}"
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

DEFAULT_EXCLUDE_FILE="${PROJECT_ROOT}/rsync_orin_exclude.txt"
if [[ -f "$DEFAULT_EXCLUDE_FILE" ]]; then
  EXCLUDE_FILES+=(--exclude-from "$DEFAULT_EXCLUDE_FILE")
fi

# 常见无需同步到设备的内容（可按需注释或改用 rsync_orin_exclude.txt）
BUILTIN_EXCLUDES=(
)

rsync -avz --delete \
  "${DRY_RUN[@]}" \
  "${BUILTIN_EXCLUDES[@]}" \
  "${EXCLUDES[@]}" \
  "${EXCLUDE_FILES[@]}" \
  -e ssh \
  "${PROJECT_ROOT}/" \
  "${ORIN_HOST}:${ORIN_DEST}/"

echo "完成: ${ORIN_HOST}:${ORIN_DEST}"
