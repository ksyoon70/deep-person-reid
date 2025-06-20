#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
Veri‑776 ↔ 신규 차량 데이터셋 정리 스크립트 (v2)
2025‑06‑20

주요 기능
---------
1) --clean 옵션: vmodel_dir, det_dir 초기화(존재 시 삭제 후 생성)
2) Veri‑776 최대 vid 분석 → kvid 시작값 산출
3) src_dir → vmodel_dir
   · src_dir : 한국차량이 모델별로 있는 폴더명
   · vsrc_dir: Veri‑776이 있는 image_train 폴더명   
   · 차량모델 폴더별로 vid → kvid 매핑 (동일 폴더 내부에서만 +1씩 증가)
   · 이미지/JSON 이름·내용 수정, 폴더 재구조화
4) vmodel_dir → det_dir
   · vmodel_dir : 차량모델 별로 이미지/JSON 저장 임시 폴더
   · dest_dir : 초종 Veri‑776에 추가로 저장 할 한국차량 데이터셋
   · **차량모델 폴더별** train/query/test 분할 (비율: 0.80/0.15/0.05)
   · 소수/0개 보정 규칙, 이미지·JSON 쌍 유지
5) 모든 단계에서 진행률 바(■ 40칸) 및 오류 메시지 출력
"""

from pathlib import Path
import shutil, json, random, argparse, sys
from typing import List, Tuple

# --------------------------------------------------
#  사용자 기본 경로 설정 (필요 시 수정)
# --------------------------------------------------
src_dir    = Path(r'E:\윤경섭\상세차종_re-id_datasets').resolve()
vsrc_dir   = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\image_train').resolve()
vmodel_dir = Path(r'E:\윤경섭\vmodel').resolve()
det_dir    = Path(r'E:\윤경섭\save').resolve()

# 분할 비율
TRAIN_RATIO = 0.80
QUERY_RATIO = 0.15
TEST_RATIO  = 0.05

BAR_WIDTH = 40  # 진행 막대 너비 ('■' 개수)

# --------------------------------------------------
#  공통 유틸리티
# --------------------------------------------------
def eprint(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)

def progress(current: int, total: int):
    done = int(BAR_WIDTH * current / total)
    bar  = '■' * done + ' ' * (BAR_WIDTH - done)
    pct  = f"{100*current/total:6.2f}%"
    print(f"\r[{bar}] {pct}", end='', flush=True)

def finish_bar():
    print(f"\r[{'■'*BAR_WIDTH}] 100.00%")

def ensure(path: Path, label: str):
    if not path.exists():
        eprint(f"{label}({path}) 없음 → 생성")
        path.mkdir(parents=True, exist_ok=True)

def require(path: Path, label: str):
    if not path.exists():
        eprint(f"{label}({path}) 없음 → 종료")
        sys.exit(1)

def is_image(p: Path):
    return p.suffix.lower() in {'.jpg', '.jpeg', '.png'}

def json_for(img: Path):
    return img.with_suffix('.json')

def extract_vid(name: str):
    try:
        return int(name.split('_', 1)[0])
    except Exception:
        return None

def next_hundred(n: int):
    return ((n // 100) + 1) * 100

def zpad(num: int, width: int=4):
    return str(num).zfill(max(width, len(str(num))))

# --------------------------------------------------
#  인자 파싱
# --------------------------------------------------
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="신규 차량 데이터셋(상세차종) → Veri 형식 변환 유틸리티"
)
parser.add_argument("-c", "--clean", action="store_true",
                    help="시작 전에 vmodel_dir, det_dir 전체를 삭제하고 새로 생성")
args = parser.parse_args()

# --------------------------------------------------
#  0. 디렉터리 준비 / 초기화
# --------------------------------------------------
require(src_dir,  "src_dir")
require(vsrc_dir, "vsrc_dir")

args.clean= True #일단 폴더 삭제하도록 함.

if args.clean:
    for d in (vmodel_dir, det_dir):
        if d.exists():
            print(f"[clean] {d} 전체 삭제 중…")
            shutil.rmtree(d)
for d in (vmodel_dir, det_dir):
    ensure(d, str(d))

# --------------------------------------------------
#  1. Veri‑776 최대 vid 파악
# --------------------------------------------------
max_vid = -1
for img in vsrc_dir.iterdir():
    if not is_image(img): continue
    vid = extract_vid(img.name)
    if vid is None:
        eprint(f"vid 추출 실패: {img.name}")
        continue
    max_vid = max(max_vid, vid)

if max_vid < 0:
    eprint("vsrc_dir 에서 유효 이미지 없음 → 종료")
    sys.exit(1)

pivot   = next_hundred(max_vid)
margin  = pivot - max_vid
kvid_seed = max_vid + margin + 1

print("=== Veri‑776 정보 ===")
print(f"  max vid  : {max_vid:04d}")
print(f"  pivot    : {pivot:04d}")
print(f"  margin   : {margin}")
print(f"  kvid seed: {kvid_seed:04d}")
print("=====================")

# --------------------------------------------------
#  2. src_dir → vmodel_dir (복사 · kvid 매핑)
# --------------------------------------------------
print("\n[1단계] src_dir → vmodel_dir 복사 시작")

files_to_copy: List[Tuple[Path, Path, Path, Path]] = []
global_kvid = kvid_seed

for folder in sorted(p for p in src_dir.rglob('*') if p.is_dir()):
    parts = folder.name.split('_')
    if len(parts) < 2:
        eprint(f"폴더명 구조 오류(스킵): {folder}")
        continue
    model, detail = parts[0], parts[1]
    dest_sub = vmodel_dir / f"{model}_{detail}"
    dest_sub.mkdir(parents=True, exist_ok=True)

    # --- 폴더별 vid→kvid 매핑 (독립적) ---
    local_vid_map = {}
    local_kvid    = global_kvid

    # 이미지 정렬 → vid 안정적 매핑
    imgs = sorted(p for p in folder.iterdir() if is_image(p))
    for img in imgs:
        jfile = json_for(img)
        if not jfile.exists():
            eprint(f"JSON 없음(스킵): {img}")
            continue
        vid = extract_vid(img.name)
        if vid is None:
            eprint(f"vid 추출 실패(스킵): {img.name}")
            continue

        # 새 kvid 결정 (폴더 내 vid 단위 +1)
        if vid not in local_vid_map:
            local_vid_map[vid] = local_kvid
            local_kvid += 1    # 같은 폴더에서만 증가
        new_vid = local_vid_map[vid]

        rest     = img.name.split('_', 1)[1]
        new_name = f"{zpad(new_vid)}_{rest}"
        dst_img  = dest_sub / new_name
        dst_json = dst_img.with_suffix('.json')
        files_to_copy.append((img, jfile, dst_img, dst_json))

    # 폴더 완료 → global_kvid 최신화
    global_kvid = local_kvid

# 실제 복사
total = len(files_to_copy)
print(f"  복사 예정 쌍: {total}")
for idx, (src_img, src_j, dst_img, dst_j) in enumerate(files_to_copy, 1):
    shutil.copy2(src_img, dst_img)

    # JSON 갱신
    with open(src_j, encoding='utf-8') as f:
        data = json.load(f)
    data['imagePath'] = dst_img.name
    with open(dst_j, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    progress(idx, total)
finish_bar()
print("\n[1단계] 완료")

# --------------------------------------------------
#  3. vmodel_dir → det_dir  (모델 폴더별 분할)
# --------------------------------------------------
print("\n[2단계] vmodel_dir → det_dir 분할 복사 시작")

train_dir = det_dir / 'image_train'
query_dir = det_dir / 'image_query'
test_dir  = det_dir / 'image_test'
for d in (train_dir, query_dir, test_dir):
    d.mkdir(parents=True, exist_ok=True)

# 전체 진행률을 위해 총 쌍 수 계산
all_pairs: List[Tuple[Path, Path]] = [
    (img, json_for(img))
    for img in vmodel_dir.rglob('*') if is_image(img)
    if json_for(img).exists()
]
grand_total = len(all_pairs)
done = 0

def copy_batch(batch: List[Tuple[Path, Path]], dest: Path):
    global done
    for src_img, src_js in batch:
        shutil.copy2(src_img, dest / src_img.name)
        shutil.copy2(src_js,  dest / src_js.name)
        done += 1
        progress(done, grand_total)

# 차량모델 폴더 단위 루프
for folder in sorted(p for p in vmodel_dir.iterdir() if p.is_dir()):
    pairs = [
        (img, json_for(img))
        for img in folder.iterdir() if is_image(img)
        if json_for(img).exists()
    ]
    if not pairs:
        continue
    random.shuffle(pairs)

    N = len(pairs)
    test_cnt  = int(N * TEST_RATIO)
    query_cnt = int(N * QUERY_RATIO)
    train_cnt = N - test_cnt - query_cnt

    # 보정
    if test_cnt == 0 and query_cnt > 0:
        query_cnt -= 1; test_cnt += 1
    elif test_cnt == 0 and train_cnt > 0:
        train_cnt -= 1; test_cnt += 1
    if query_cnt == 0 and train_cnt > 0:
        train_cnt -= 1; query_cnt += 1

    assert test_cnt + query_cnt + train_cnt == N

    # 순서: test → query → train
    copy_batch(pairs[:test_cnt],                  test_dir)
    copy_batch(pairs[test_cnt:test_cnt+query_cnt], query_dir)
    copy_batch(pairs[test_cnt+query_cnt:],         train_dir)

finish_bar()
print("\n[2단계] 완료")
print("\n=== 모든 작업이 정상적으로 끝났습니다 ===")
