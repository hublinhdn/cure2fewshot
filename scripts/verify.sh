#!/usr/bin/env bash
# verify.sh — doi chieu ket qua cua ban voi expected/.
#
# So tung file JSON do mot lan chay sinh ra voi ban dong bang trong expected/, BO QUA dung cac
# truong ghi duong dan tuyet doi cua may da sinh ra so (manifest, report, sam_ckpt). Khong bo qua
# bat ky gia tri do duoc nao. So nguyen, chuoi va diem rubric phai bang nhau tuyet doi; so thuc
# duoc so voi dung sai FLOAT_TOL (mac dinh 1e-6), vi trich embedding lai tren chinh may tham chieu
# da cho lech co 1e-7 o vai thong ke cosine ma khong doi bat ky diem rubric nao.
#
# Chay tu goc repo, sau khi da lam theo REPRODUCE.md:
#   bash scripts/verify.sh
#
# Ma thoat: 0 neu moi file khop, 1 neu co file lech, 2 neu thieu file can so.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
OUT="${CURE_FITNESS_OUT:-$PWD/outputs}"
OUT_V2="${PHASE_F_V2_OUT:-$PWD/outputs_v2}"

PY="${PYTHON:-python3}"
IGNORE_KEYS="manifest,report,sam_ckpt,out_dir,crops_root,raw_root,gallery"
FLOAT_TOL="${FLOAT_TOL:-1e-6}"

# expected/<ten>.json  <=>  duong dan file do lan chay sinh ra
PAIRS=(
  "cure_processed_summary.json|$OUT/cure_processed_summary.json"
  "phaseD_log.json|$OUT/phaseD_log.json"
  "removal_log.json|$OUT/removal_log.json"
  "fitness_report_direct.json|$OUT/fitness_report.json"
  "rubric_score_direct.json|$OUT/rubric_score_direct.json"
  "hardened_log_v2.json|$OUT_V2/hardened_v2_log.json"
  "fitness_report_v2.json|$OUT_V2/fitness_v2/fitness_report.json"
  "rubric_score_v2.json|$OUT_V2/rubric_v2/rubric_score_v2.json"
)

echo "verify.sh — doi chieu voi expected/"
echo "  outputs      = $OUT"
echo "  outputs_v2   = $OUT_V2"
echo "  bo qua truong: $IGNORE_KEYS"
echo "  dung sai so thuc: $FLOAT_TOL (so nguyen, chuoi va diem rubric: bang tuyet doi)"
echo

fail=0; missing=0; ok=0

for pair in "${PAIRS[@]}"; do
  exp="expected/${pair%%|*}"
  got="${pair##*|}"
  name="${pair%%|*}"
  if [ ! -f "$exp" ]; then
    printf '  %-34s THIEU BAN DONG BANG (%s)\n' "$name" "$exp"; missing=$((missing+1)); continue
  fi
  if [ ! -f "$got" ]; then
    printf '  %-34s CHUA CO KET QUA (%s)\n' "$name" "$got"; missing=$((missing+1)); continue
  fi
  if "$PY" - "$exp" "$got" "$IGNORE_KEYS" "$FLOAT_TOL" <<'EOF'
import json, sys

def strip(o, keys):
    if isinstance(o, dict):
        return {k: strip(v, keys) for k, v in o.items() if k not in keys}
    if isinstance(o, list):
        return [strip(v, keys) for v in o]
    return o

def same(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= tol
    return a == b

exp_p, got_p, ign, tol = sys.argv[1], sys.argv[2], set(sys.argv[3].split(",")), float(sys.argv[4])
a = strip(json.load(open(exp_p)), ign)
b = strip(json.load(open(got_p)), ign)
sys.exit(0 if same(a, b, tol) else 1)
EOF
  then
    printf '  %-34s KHOP\n' "$name"; ok=$((ok+1))
  else
    printf '  %-34s LECH\n' "$name"
    "$PY" - "$exp" "$got" "$IGNORE_KEYS" "$FLOAT_TOL" <<'EOF'
import json, sys

def flat(o, p="", out=None):
    out = {} if out is None else out
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, f"{p}.{k}" if p else k, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            flat(v, f"{p}[{i}]", out)
    else:
        out[p] = o
    return out

ign, tol = set(sys.argv[3].split(",")), float(sys.argv[4])
a = flat(json.load(open(sys.argv[1])))
b = flat(json.load(open(sys.argv[2])))
n = 0
for k in sorted(set(a) | set(b)):
    if k.split(".")[-1].split("[")[0] in ign:
        continue
    x, y = a.get(k, "<thieu>"), b.get(k, "<thieu>")
    if (isinstance(x, float) or isinstance(y, float)) and isinstance(x, (int, float)) \
            and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool) \
            and abs(x - y) <= tol:
        continue
    if x != y:
        print(f"      {k}: expected={x!r} got={y!r}")
        n += 1
        if n >= 12:
            print("      ...")
            break
EOF
    fail=$((fail+1))
  fi
done

echo
echo "khop=$ok  lech=$fail  thieu=$missing"
if [ "$missing" -gt 0 ] && [ "$fail" -eq 0 ]; then
  echo "Chua chay du cac buoc. Xem REPRODUCE.md."
  exit 2
fi
if [ "$fail" -gt 0 ]; then
  echo "CO FILE LECH. Neu chi lech o so pixel cua crop thi kiem phien ban Pillow (ghim 12.0.0);"
  echo "neu lech o gia tri do duoc thi bao loi kem log nay."
  exit 1
fi
echo "TAT CA KHOP."
