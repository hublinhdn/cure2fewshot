#!/usr/bin/env bash
# verify.sh — doi chieu ket qua cua ban voi expected/.
#
# So tung file JSON do mot lan chay sinh ra voi ban dong bang trong expected/, BO QUA dung cac
# truong ghi duong dan tuyet doi cua may da sinh ra so (manifest, report, sam_ckpt, ...) va truong
# `note` cua rubric (chuoi mo ta cho nguoi doc, chua so do da lam tron 3 chu so; diem cua tung
# tieu chi van duoc so). Khong bo qua bat ky gia tri do duoc nao. So nguyen, chuoi va diem rubric
# phai bang nhau tuyet doi; so thuc duoc so voi dung sai FLOAT_TOL, mac dinh 1e-5.
#
# Dung sai do tu hai phep do (REPRODUCE.md muc 7): trich embedding lai tren chinh may tham chieu
# (Apple silicon, MPS) cho lech toi da 5.4e-7; mot ban clone sach tren may CUDA (RTX 3080), voi
# TF32 tat nhu ma tu 1.0.1, cho lech toi da 1.3e-6 (percentile cosine cua ViT). Mot query doi
# hang lam mot control doi 1.2e-4, nen 1e-5 tach nhieu so hoc khoi ket qua doi.
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
IGNORE_KEYS="manifest,report,sam_ckpt,out_dir,crops_root,raw_root,gallery,note"
FLOAT_TOL="${FLOAT_TOL:-1e-5}"

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

fail_float=0; fail_hard=0; missing=0; ok=0

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
  # Dong dau: "OK <lech so thuc toi da>" hoac "MISMATCH <lech so thuc toi da> <so truong khac so thuc lech>",
  # sau do toi da 12 dong chi tiet.
  res="$("$PY" - "$exp" "$got" "$IGNORE_KEYS" "$FLOAT_TOL" <<'PYEOF'
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

def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

ign, tol = set(sys.argv[3].split(",")), float(sys.argv[4])
a = flat(json.load(open(sys.argv[1])))
b = flat(json.load(open(sys.argv[2])))
max_float, hard, lines = 0.0, 0, []
for k in sorted(set(a) | set(b)):
    if k.split(".")[-1].split("[")[0] in ign:
        continue
    x, y = a.get(k, "<thieu>"), b.get(k, "<thieu>")
    if (isinstance(x, float) or isinstance(y, float)) and is_num(x) and is_num(y):
        d = abs(x - y)
        max_float = max(max_float, d)
        if d <= tol:
            continue
        lines.append(f"      {k}: expected={x!r} got={y!r} (lech {d:.3e})")
    elif x != y:
        hard += 1
        lines.append(f"      {k}: expected={x!r} got={y!r}")
if lines:
    print(f"MISMATCH {max_float:.3e} {hard}")
    print("\n".join(lines[:12]))
    if len(lines) > 12:
        print("      ...")
    sys.exit(1)
print(f"OK {max_float:.3e}")
PYEOF
)"
  head1="${res%%$'\n'*}"
  set -- $head1
  if [ "$1" = "OK" ]; then
    printf '  %-34s KHOP   (lech so thuc toi da %s)\n' "$name" "$2"; ok=$((ok+1))
  else
    printf '  %-34s LECH   (lech so thuc toi da %s, truong khac so thuc lech: %s)\n' "$name" "$2" "$3"
    printf '%s\n' "${res#*$'\n'}"
    if [ "$3" -gt 0 ]; then fail_hard=$((fail_hard+1)); else fail_float=$((fail_float+1)); fi
  fi
done

fail=$((fail_float+fail_hard))
echo
echo "khop=$ok  lech=$fail  thieu=$missing"
if [ "$missing" -gt 0 ] && [ "$fail" -eq 0 ]; then
  echo "Chua chay du cac buoc. Xem REPRODUCE.md."
  exit 2
fi
if [ "$fail_hard" -gt 0 ]; then
  echo "CO FILE LECH o so nguyen, chuoi, co hoac diem rubric: day la ket qua khac, khong phai nhieu so hoc."
  echo "Kiem lai thu tu cac buoc trong REPRODUCE.md (nhat la muc 6), roi bao loi kem log nay."
  exit 1
fi
if [ "$fail_float" -gt 0 ]; then
  echo "CO FILE LECH chi o so thuc do duoc, vuot dung sai FLOAT_TOL=$FLOAT_TOL, moi so nguyen va diem rubric van khop."
  echo "Lech co 1e-4 den 1e-3 tap trung o cac truong resnet50 la dau hieu embedding trich bang TF32"
  echo "(GPU Ampere tro len voi ma truoc 1.0.1, hoac cudnn.allow_tf32 bi bat lai): trich lai embedding"
  echo "bang ma hien tai. Lech nho hon ma vuot dung sai: ghi phien ban torch, timm, GPU va bao loi kem log nay."
  exit 1
fi
echo "TAT CA KHOP."
