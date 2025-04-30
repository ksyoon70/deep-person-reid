#VeRi\veri\image_add 에 labelme에서 라벨링한 dataset을 넣어 놓으면
#VeRi\veri\image_train 에서 최대 id를 읽어서
#Veri\veri\result 에 최대 id 이후로 id를 바꾸어 저장하는 함수이다.
import os
import json
import shutil
from typing import Set   # Python 3.8 호환 타입 힌트
from glob import glob

# ────────────────────────────────
# 1.  색·종류 레퍼런스 읽기
# ────────────────────────────────
def load_first_tokens(txt_path: str) -> Set[str]:
    """각 라인의 첫 번째 토큰(token\tid 형식)을 집합으로 반환"""
    tokens: Set[str] = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.add(line.split()[0])      # 탭·공백 모두 split 대응
    return tokens

# 프로젝트 루트(필요 시 절대경로 지정)
BASE_DIR   = os.path.join("VeRi", "veri")

COLOR_SET = load_first_tokens(os.path.join(BASE_DIR, "list_color.txt"))
TYPE_SET  = load_first_tokens(os.path.join(BASE_DIR, "list_type.txt"))

DEST_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(DEST_DIR, exist_ok=True)


# ────────────────────────────────
# 1-A.  image_train 폴더의 최대 ID 찾기
# ────────────────────────────────
def find_max_numeric_prefix(img_root: str) -> int:
    """'image_train' 하위 모든 이미지 중, 이름을 '_' 로 나눴을 때
    첫 파트가 숫자인 경우의 최대값을 반환한다.
    (숫자 파일이 없으면 0 반환)
    """
    img_patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp"]
    max_id = 0
    for pattern in img_patterns:
        for path in glob(os.path.join(img_root, pattern), recursive=True):
            name = os.path.basename(path)
            first_part = name.split("_")[0]
            if first_part.isdigit():
                max_id = max(max_id, int(first_part))
    return max_id

# ────────────────────────────────
# 2.  JSON 1개 처리
# ────────────────────────────────
def process_one_json(json_path: str, dest_dir: str, max_id: int) -> int:
    """json_path를 변환·저장하고, 사용한 id를 반환"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # ---------- label 정리 ----------
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        parts = label.split("_")
        if len(parts) == 2:
            vtype, color = parts          # 형식: type_color
            if color in COLOR_SET and vtype in TYPE_SET:
                shape["label"] = vtype
                shape["color"] = color

    # ---------- 파일명(ID) 갱신 ----------
    base_name = os.path.basename(json_path)           # 0001_xxx.json
    stem, _ = os.path.splitext(base_name)             # 0001_xxx
    name_parts = stem.split("_")

    new_id = max_id
    if name_parts[0].isdigit():                       # 숫자 ID가 있을 때만
        width   = len(name_parts[0])                  # 0001 → 4자리 유지
        new_id  = max_id + 1
        id_str  = str(new_id).zfill(width)
        name_parts[0] = id_str

        new_stem       = "_".join(name_parts)
        new_json_name  = new_stem + ".json"

        # imagePath 수정
        img_name = data.get("imagePath", "")
        if img_name:
            img_stem, img_ext = os.path.splitext(img_name)
            img_parts         = img_stem.split("_")
            if img_parts and img_parts[0].isdigit():
                img_parts[0]   = id_str
                new_img_name   = "_".join(img_parts) + img_ext
                data["imagePath"] = new_img_name
            else:
                new_img_name = img_name
        else:
            new_img_name = ""
    else:                                            # 숫자 ID가 없으면 그대로
        new_json_name  = base_name
        new_img_name   = data.get("imagePath", "")

    # ---------- JSON 저장 ----------
    dst_json_path = os.path.join(dest_dir, new_json_name)
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ---------- 이미지 복사 ----------
    if new_img_name:
        src_img_path = os.path.join(os.path.dirname(json_path),img_name)
        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path,
            os.path.join(dest_dir, new_img_name))

    return new_id      # 갱신된 id 리턴 (변경 없으면 기존 max_id 그대로)

# ────────────────────────────────
# 3.  실행
# ────────────────────────────────
def main() -> None:

    # 3-A. image_train 최대 ID 계산
    train_dir = os.path.join(BASE_DIR, "image_train")
    max_id = find_max_numeric_prefix(train_dir)
    print(f"image_train 폴더에서 발견한 최대 ID: {max_id}")

    bar_len = 40
    src_json_dir = os.path.join(BASE_DIR, "image_add")
    # --- 처리 대상 전체 목록(재귀) 한꺼번에 수집 ---
    all_json = glob(os.path.join(src_json_dir, '**', '*.json'), recursive=True)
    total = len(all_json)
    if total == 0:
        print('변환할 파일이 없습니다.')
        exit(0)

    # --- 변환 루프 ---
    for idx, json_path in enumerate(all_json, 1):      # 1-based
        max_id = process_one_json(json_path, DEST_DIR, max_id)
        # 진행상황 막대그래프
        done = int(bar_len * idx / total)
        bar = '■' * done + '-' * (bar_len - done)
        print(f'진행중: [{bar}] {idx}/{total}', end='\r')

    print(f"완료: '{DEST_DIR}' 폴더에 변환-복사된 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()
