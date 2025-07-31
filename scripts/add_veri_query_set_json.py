#!/usr/bin/env python3
# -*- coding: utf‑8 -*-
"""
Veri‑776에서 query set에 JSON을 추가하는 스크립트입니다.
이 스크립트는 Veri-776 데이터셋의 쿼리 이미지에 대한 JSON 파일을 생성합니다.
2025‑7‑31
작성자: 윤경섭

주요 기능
---------
query 이미지 디렉토리와 갤러리 이미지 디렉토리에 json이 없다면, 같은 파일이 gallery 디렉토리에 있는지 확인합니다.
있다면 gallery 디렉토리에 있는 json 파일을 query 디렉토리에 복사합니다.
"""

from pathlib import Path
import shutil, json, random, argparse, sys
from typing import List, Tuple

# --------------------------------------------------
#  사용자 기본 경로 설정 (필요 시 수정)
# --------------------------------------------------
# query 이미지 경로
query_dir    = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\image_query').resolve()
# gallery 이미지 경로
gallery_dir   = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\image_test').resolve()

# --------------------------------------------------
#  (추가) query_dir ↔ gallery_dir : 누락된 JSON 복사
# --------------------------------------------------
BAR_WIDTH = 40
IMG_EXTS  = {'.jpg', '.jpeg', '.png'}

def is_image_file(p: Path):
    return p.suffix.lower() in IMG_EXTS

def progress_bar(current: int, total: int):
    done  = int(BAR_WIDTH * current / total) if total else BAR_WIDTH
    bar   = '■' * done + ' ' * (BAR_WIDTH - done)
    pct   = f"{100 * current / total:6.2f}%" if total else "100.00%"
    print(f"\r[{bar}] {pct}", end='', flush=True)

# 1) 복사 대상 목록 수집
tasks: List[Tuple[Path, Path]] = []          # (src_json , dst_json)
for img_path in query_dir.iterdir():
    if not is_image_file(img_path):
        continue
    json_q = img_path.with_suffix('.json')
    if json_q.exists():
        continue                             # 이미 JSON 있음 → 패스

    # gallery_dir 에 동명 JSON 존재?
    json_g = gallery_dir / json_q.name
    if json_g.exists():
        tasks.append((json_g, json_q))

total_tasks = len(tasks)
if total_tasks == 0:
    print("[INFO] query_dir 의 모든 이미지에 JSON 이 이미 존재합니다.")
else:
    print(f"[INFO] 누락된 JSON {total_tasks}개를 gallery → query 로 복사합니다.")

    # 2) 복사 수행 + 진행 막대
    for i, (src_json, dst_json) in enumerate(tasks, 1):
        shutil.copy2(src_json, dst_json)
        print(f"\n[COPY] {src_json.name} → query_dir")   # 줄바꿈으로 COPY 로그
        progress_bar(i, total_tasks)

    # 진행 막대를 100%로 맞추고 줄바꿈
    progress_bar(total_tasks, total_tasks)
    print("\n[INFO] JSON 복사가 완료되었습니다.")


print("\n=== 모든 작업이 정상적으로 끝났습니다 ===")
