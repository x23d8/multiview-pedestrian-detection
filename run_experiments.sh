#!/usr/bin/env bash
# Chạy ma trận thực nghiệm: model x drop x loss x pseudo-label. Mỗi tổ hợp gọi `run.sh` một lần.
#
#   bash run_experiments.sh                                   # 5 model, drop 0/20/45/60, orig+aabl, không pseudo
#   bash run_experiments.sh --models mvdet,shot --drops 60 --losses aabl --pseudo both
#   bash run_experiments.sh --drop-mode frame --pseudo on --data-path /data/Wildtrack
#   bash run_experiments.sh --dry-run                         # chỉ in lệnh
#
# Tên loss: orig = loss gốc (MVDet/SHOT: mse, MVDeTr/EarlyBird/TrackTacular: focal); aabl = `brl` trong code.
# drop 0 = full supervised. Pseudo-label không áp dụng cho drop 0 (không có nhãn bị thiếu) nên bị bỏ qua.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

ALL_MODELS=(mvdet mvdetr shot earlybird tracktacular)
MODELS=("${ALL_MODELS[@]}")
DROPS=(0 20 45 60)
LOSSES=(orig aabl)
PSEUDO=off                 # off | on | both
DROP_MODE=instance         # instance | frame
DATASET=wildtrack          # wildtrack | multiviewx | wavetrack (wavetrack: chạy data_tools/prepare_wavetrack.py trước)
DATA_PATH=""
CACHE=""
GPU=""
EPOCHS=""
SEED=""
PSEUDO_WEIGHT=""           # mặc định: 0.01 khi drop-mode=frame, còn lại dùng mặc định của model
SHOT_ORIG_LOSS=mse         # mse | shot_mse
YOLO_WEIGHTS=yolo26s.pt
GEN_CACHE=1
GEN_DROPS=1
TEST_AFTER=1
SKIP_DONE=1
DRY=0
COLLECT=""            # thư mục gom log/test/weight của từng run (vd. Kaggle: /kaggle/working/output)
SAVE_WEIGHTS=1
EXTRA=()

usage() { sed -n '2,12p' "$0"; cat <<'EOF'

Tuỳ chọn:
  --models a,b,..      mvdet,mvdetr,shot,earlybird,tracktacular (mặc định: cả 5)
  --drops 0,20,45,60   mức drop; 0 = full supervised
  --losses orig,aabl   orig = loss gốc của model, aabl = brl
  --pseudo off|on|both tắt / bật / chạy cả hai
  --drop-mode M        instance (drop_XX) | frame (frame_drop_XX)
  --dataset D          wildtrack | multiviewx | wavetrack
  --data-path P        thư mục dataset (mặc định: <repo>/Data/<Wildtrack|MultiviewX>)
  --cache FILE         pseudo cache JSON (mặc định: <Data>/pseudo_cache/<dataset>_yolo26s.json)
  --weights W          weights YOLO khi sinh cache (mặc định yolo26s.pt)
  --pseudo-weight X    pseudo_loss_weight, bật kèm peak loss (mặc định 0.01 nếu drop-mode=frame)
  --shot-loss L        loss gốc của SHOT: mse | shot_mse (mặc định mse)
  --gpu ID             CUDA_VISIBLE_DEVICES
  --epochs N           số epoch (tất cả model)
  --seed N             seed (chỉ MVDet/MVDeTr/SHOT)
  --no-gen-cache       không tự sinh pseudo cache nếu thiếu
  --no-gen-drops       không tự sinh nhãn drop nếu thiếu
  --no-test            không tự test last.ckpt sau khi train EarlyBird/TrackTacular
  --collect DIR        sau mỗi run, copy log.txt/test/config/weight vào DIR/<tag>/
  --no-weights         với --collect: bỏ file weight (*.pth, *.ckpt)
  --rerun              chạy lại cả các tổ hợp đã xong
  --dry-run            chỉ in lệnh
  -- ...               mọi tham số sau `--` được thêm vào cuối mỗi lệnh train
EOF
}

csv() { local IFS=,; read -ra "$1" <<<"$2"; }
die() { echo "[exp] LỖI: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case $1 in
        --models) csv MODELS "$2"; shift 2 ;;
        --drops) csv DROPS "$2"; shift 2 ;;
        --losses) csv LOSSES "$2"; shift 2 ;;
        --pseudo) PSEUDO=$2; shift 2 ;;
        --drop-mode) DROP_MODE=$2; shift 2 ;;
        --dataset) DATASET=$2; shift 2 ;;
        --data-path) DATA_PATH=$2; shift 2 ;;
        --cache) CACHE=$2; shift 2 ;;
        --weights) YOLO_WEIGHTS=$2; shift 2 ;;
        --pseudo-weight) PSEUDO_WEIGHT=$2; shift 2 ;;
        --shot-loss) SHOT_ORIG_LOSS=$2; shift 2 ;;
        --gpu) GPU=$2; shift 2 ;;
        --epochs) EPOCHS=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        --no-gen-cache) GEN_CACHE=0; shift ;;
        --no-gen-drops) GEN_DROPS=0; shift ;;
        --no-test) TEST_AFTER=0; shift ;;
        --collect) COLLECT=$2; shift 2 ;;
        --no-weights) SAVE_WEIGHTS=0; shift ;;
        --rerun) SKIP_DONE=0; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) usage; die "tham số lạ: $1" ;;
    esac
done

# ---- kiểm tra tham số ----
for m in "${MODELS[@]}"; do
    [[ " ${ALL_MODELS[*]} " == *" $m "* ]] || die "model không hợp lệ: $m"
done
for d in "${DROPS[@]}"; do [[ $d =~ ^(0|20|45|60)$ ]] || die "drop phải là 0/20/45/60, nhận $d"; done
for l in "${LOSSES[@]}"; do [[ $l == orig || $l == aabl ]] || die "loss phải là orig hoặc aabl, nhận $l"; done
[[ $PSEUDO =~ ^(off|on|both)$ ]] || die "--pseudo phải là off|on|both"
[[ $DROP_MODE =~ ^(instance|frame)$ ]] || die "--drop-mode phải là instance|frame"
[[ $DATASET =~ ^(wildtrack|multiviewx|wavetrack)$ ]] || die "--dataset phải là wildtrack|multiviewx|wavetrack"
[[ $SHOT_ORIG_LOSS =~ ^(mse|shot_mse)$ ]] || die "--shot-loss phải là mse|shot_mse"

case $DATASET in wildtrack) DS_DIR=Wildtrack ;; multiviewx) DS_DIR=MultiviewX ;; wavetrack) DS_DIR=WaveTrackNew ;; esac
DATA_PATH=${DATA_PATH:-$ROOT/Data/$DS_DIR}
# run.sh cd vào thư mục model, nên mọi đường dẫn phải là tuyệt đối
if [ -d "$DATA_PATH" ]; then DATA_PATH=$(cd "$DATA_PATH" && pwd); elif [ $DRY -eq 0 ]; then die "không thấy dataset: $DATA_PATH"; fi
DATA_ROOT=$(dirname "$DATA_PATH")
DATA_NAME=$(basename "$DATA_PATH")
CACHE=${CACHE:-$DATA_ROOT/pseudo_cache/${DATASET}_yolo26s.json}
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU

if [ -z "$PSEUDO_WEIGHT" ] && [ "$DROP_MODE" = frame ]; then PSEUDO_WEIGHT=0.01; fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/experiment_logs
mkdir -p "$OUT/done" "$OUT/runs"
SUMMARY=$OUT/summary_$STAMP.tsv
echo -e "tag\tstatus\tseconds" > "$SUMMARY"

log() { echo "[exp] $*"; }
run() { if [ $DRY -eq 1 ]; then printf '  '; printf '%q ' "$@"; echo; else "$@"; fi; }

# ---- chuẩn bị dữ liệu dùng chung ----
need_pseudo=0
[ "$PSEUDO" != off ] && need_pseudo=1
if [ $need_pseudo -eq 1 ] && [ ! -f "$CACHE" ]; then
    if [ $GEN_CACHE -eq 1 ]; then
        log "thiếu pseudo cache, sinh bằng YOLO: $CACHE"
        run bash "$ROOT/run.sh" tools data_tools/generate_pseudo_cache.py --dataset "$DATASET" \
            --data_root "$DATA_PATH" --output "$CACHE" --weights "$YOLO_WEIGHTS" ${GPU:+--device "$GPU"} \
            || die "sinh pseudo cache thất bại"
    else
        die "thiếu pseudo cache: $CACHE (bỏ --no-gen-cache để tự sinh)"
    fi
fi

drop_dir_prefix=drop; [ "$DROP_MODE" = frame ] && drop_dir_prefix=frame_drop
missing_drops=()
for d in "${DROPS[@]}"; do
    [ "$d" = 0 ] && continue
    [ -d "$DATA_PATH/drop_annotations/${drop_dir_prefix}_$d" ] || missing_drops+=("drop$d")
done
if [ ${#missing_drops[@]} -gt 0 ]; then
    if [ $GEN_DROPS -eq 1 ]; then
        log "thiếu nhãn drop (${missing_drops[*]}), đang sinh (mode=$DROP_MODE)"
        run bash "$ROOT/run.sh" tools data_tools/simulate_dropped_annotations.py -d "$DATA_NAME" -r "$DATA_ROOT" \
            -s "${missing_drops[@]}" --mode "$DROP_MODE" --seed 1 || die "sinh nhãn drop thất bại"
    else
        die "thiếu nhãn drop: ${missing_drops[*]} (bỏ --no-gen-drops để tự sinh)"
    fi
fi

# ---- dựng lệnh train cho 1 tổ hợp ----
build_cmd() { # model drop loss pseudo(0/1) -> CMD
    local m=$1 d=$2 l=$3 pl=$4 loss
    CMD=()
    case $m in
        mvdet|shot|mvdetr)
            if [ "$l" = aabl ]; then loss=brl
            elif [ "$m" = mvdet ]; then loss=mse
            elif [ "$m" = shot ]; then loss=$SHOT_ORIG_LOSS
            else loss=focal; fi
            CMD=(-d "$DATASET" --data_path "$DATA_PATH" --drop_ratio "$d" --loss "$loss")
            [ "$d" != 0 ] && [ "$DROP_MODE" = frame ] && CMD+=(--drop_mode frame)
            if [ "$pl" = 1 ]; then
                CMD+=(--use_pseudo_labels --pseudo_cache "$CACHE")
                [ -n "$PSEUDO_WEIGHT" ] && CMD+=(--pseudo_peak_loss --pseudo_loss_weight "$PSEUDO_WEIGHT")
            fi
            [ -n "$EPOCHS" ] && CMD+=(--epochs "$EPOCHS")
            [ -n "$SEED" ] && CMD+=(--seed "$SEED")
            ;;
        earlybird|tracktacular)
            if [ "$l" = aabl ]; then loss=brl; else loss=focal; fi
            CMD=(fit -c configs/t_fit.yml -c "configs/d_$DATASET.yml")
            [ "$m" = tracktacular ] && CMD+=(-c configs/m_mvdet.yml)
            CMD+=(--data.init_args.data_dir "$DATA_PATH" --data.init_args.drop_ratio "$d" --model.heatmap_loss "$loss")
            [ "$d" != 0 ] && [ "$DROP_MODE" = frame ] && CMD+=(--data.init_args.drop_mode frame)
            if [ "$pl" = 1 ]; then
                CMD+=(--data.init_args.pseudo_cache "$CACHE" --model.use_pseudo_labels true)
                [ -n "$PSEUDO_WEIGHT" ] && CMD+=(--model.pseudo_no_peak_loss false --model.pseudo_loss_weight "$PSEUDO_WEIGHT")
            fi
            [ -n "$EPOCHS" ] && CMD+=(--trainer.max_epochs "$EPOCHS")
            ;;
    esac
    CMD+=("${EXTRA[@]}")
}

latest_version() { # model_dir -> số version lớn nhất (hoặc -1)
    local best=-1 v
    for p in "$1"/lightning_logs/version_*; do
        [ -d "$p" ] || continue
        v=${p##*version_}; [[ $v =~ ^[0-9]+$ ]] && [ "$v" -gt "$best" ] && best=$v
    done
    echo "$best"
}

model_rel() {
    case $1 in
        mvdet) echo models/MVDet ;;
        mvdetr) echo models/MVDeTr ;;
        shot) echo models/shot ;;
    esac
}

# gom artifact của 1 run vào $COLLECT/<tag>: lệnh, console, log.txt/test.txt/weight (MVDet, MVDeTr, SHOT)
# hoặc các lightning_logs/version_N mới (EarlyBird, TrackTacular: gồm cả version do bước test tạo ra)
collect_run() {
    local m=$1 tag=$2 marker=$3 wd=$4 before=$5 dest="$COLLECT/$tag" f src v p
    mkdir -p "$dest"
    cp -f "$OUT/runs/$STAMP.$tag.log" "$dest/console.log" 2>/dev/null
    { printf '%q ' bash run.sh "$m" "${CMD[@]}"; echo; } > "$dest/command.txt"
    if [ -n "$wd" ]; then
        for p in "$wd"/lightning_logs/version_*; do
            [ -d "$p" ] || continue
            v=${p##*version_}
            [[ $v =~ ^[0-9]+$ ]] && [ "$v" -gt "$before" ] && cp -r "$p" "$dest/lightning_logs_version_$v"
        done
    else
        while IFS= read -r f; do
            src=$(dirname "$f")
            cp -r "$src" "$dest/$(basename "$src")"
            rm -rf "$dest/$(basename "$src")/scripts"   # bản copy mã nguồn do main.py tạo
        done < <(find "$ROOT/$(model_rel "$m")/logs" -type f -name log.txt -newer "$marker" 2>/dev/null)
    fi
    if [ "$SAVE_WEIGHTS" -eq 0 ]; then
        find "$dest" -type f \( -name '*.pth' -o -name '*.ckpt' -o -name '*.pt' \) -delete
    fi
}

model_workdir() {
    case $1 in
        earlybird) echo "$ROOT/models/EarlyBird/EarlyBird" ;;
        tracktacular) echo "$ROOT/models/TrackTacular/WorldTrack" ;;
    esac
}

# ---- vòng lặp thực nghiệm ----
pl_values=(0)
[ "$PSEUDO" = on ] && pl_values=(1)
[ "$PSEUDO" = both ] && pl_values=(0 1)
total=0; failed=0

for m in "${MODELS[@]}"; do
  for d in "${DROPS[@]}"; do
    for l in "${LOSSES[@]}"; do
      for pl in "${pl_values[@]}"; do
        if [ "$pl" = 1 ] && [ "$d" = 0 ]; then
            log "bỏ qua $m drop0 pseudo (full supervised không có nhãn thiếu)"; continue
        fi
        tag="${DATASET}_${m}_${DROP_MODE}$d"; [ "$d" = 0 ] && tag="${DATASET}_${m}_full"
        tag="${tag}_${l}_$([ "$pl" = 1 ] && echo pl || echo nopl)"
        if [ $SKIP_DONE -eq 1 ] && [ $DRY -eq 0 ] && [ -f "$OUT/done/$tag" ]; then
            log "đã xong, bỏ qua: $tag"; continue
        fi
        build_cmd "$m" "$d" "$l" "$pl"
        total=$((total+1))
        log "=== $tag ==="
        t0=$(date +%s); status=ok
        if [ $DRY -eq 1 ]; then
            run bash "$ROOT/run.sh" "$m" "${CMD[@]}"
            [ "$TEST_AFTER" = 1 ] && [[ $m == earlybird || $m == tracktacular ]] && echo "  (sau train: test lightning_logs/version_N/checkpoints/last.ckpt)"
            continue
        fi
        wd=""; before=-1
        marker=$OUT/.marker_$tag; touch "$marker"
        if [[ $m == earlybird || $m == tracktacular ]]; then wd=$(model_workdir "$m"); before=$(latest_version "$wd"); fi
        bash "$ROOT/run.sh" "$m" "${CMD[@]}" 2>&1 | tee "$OUT/runs/$STAMP.$tag.log"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -ne 0 ]; then
            status=train_fail
        elif [ -n "$wd" ] && [ "$TEST_AFTER" = 1 ]; then
            after=$(latest_version "$wd")
            if [ "$after" -le "$before" ]; then
                status=no_version
            else
                ver=lightning_logs/version_$after
                log "test $m: $ver (last.ckpt)"
                bash "$ROOT/run.sh" "$m" test -c "$ver/config.yaml" --ckpt_path "$ver/checkpoints/last.ckpt" \
                    2>&1 | tee -a "$OUT/runs/$STAMP.$tag.log"
                [ "${PIPESTATUS[0]}" -ne 0 ] && status=test_fail
                echo "$tag -> $wd/$ver" >> "$OUT/versions.txt"
            fi
        fi
        [ -n "$COLLECT" ] && collect_run "$m" "$tag" "$marker" "$wd" "$before"
        [ "$status" = ok ] && touch "$OUT/done/$tag" || failed=$((failed+1))
        echo -e "$tag\t$status\t$(( $(date +%s) - t0 ))" >> "$SUMMARY"
        log "$tag: $status"
      done
    done
  done
done

log "xong: $total lần chạy, $failed lỗi"
[ $DRY -eq 0 ] && { log "tổng kết: $SUMMARY"; column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"; }
[ "$failed" -eq 0 ]
