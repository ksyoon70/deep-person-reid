#VeRi\veri\image_add 에 labelme에서 라벨링한 dataset을 넣어 놓으면
#VeRi\veri\image_train 에서 최대 id를 읽어서
#Veri\veri\result 에 최대 id 이후로 id를 바꾸어 저장하는 함수이다.
import os
import json
import shutil
from typing import Set   # Python 3.8 호환 타입 힌트
from glob import glob
from pathlib import Path
from collections import Counter, defaultdict
from natsort import natsorted   # 사람이 읽기 좋은 자연 정렬(선택 사항)
from typing import Iterator

# --- 0) 4자리 숫자 PID를 담을 리스트 -----------------------------
pid_list = []          # 필요하면 set() 후 나중에 리스트 변환도 가능
pid_seen = set()       # 중복 방지용 내부 집합

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
def process_one_json(json_path: str, dest_dir: str, pid_list: list, pid_seen : set) -> None:
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

    if name_parts[0].isdigit():
        pid = int(name_parts[0])  # 숫자 ID가 있을 때만
        if pid in pid_seen:
            new_id = (pid_list[-1] if pid_list else 1)
            if not pid_list:
                pid_list.add(1) 
        else:
            max_id = (pid_list[-1] if pid_list else 0) + 1                  # 0001 → 4자리 유지
            new_id  = max_id
            pid_seen.add(pid)
            pid_list.append(max_id)
        width   = len(name_parts[0])
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

    return

def scan_folders(root: Path) -> Iterator[Path]:
    """root 이하 모든 폴더를 yield.
    하위 폴더가 없으면 root 자체를 한 번 yield.
    """
    root = root.resolve()
    yielded = False

    # os.walk() 는 내부적으로 os.scandir()를 사용 → DirEntry.is_dir() 캐시로 빠름
    for dirpath, dirnames, _ in os.walk(root):
        for dn in dirnames:          # 파일 목록(filenames)은 무시
            yielded = True
            yield Path(dirpath) / dn

    if not yielded:                 # 하위 폴더가 하나도 없었다면
        yield root

def iter_json_sorted(folder: Path, *, natural=False):
    """folder 안의 *.json 파일을 이름 기준으로 정렬해 yield"""
    json_paths = list(folder.glob('*.json'))

    # ① 알파벳/사전 순
    if not natural:
        json_paths = sorted(json_paths, key=lambda p: p.name)        # 또는 p.stem

    # ② 사람이 읽기 좋은 자연 정렬(파일1, 파일2, … 파일10)
    else:
        json_paths = natsorted(json_paths, key=lambda p: p.name)

    for p in json_paths:
        yield p


def count_json_per_folder(root_dir):
    """
    root_dir 이하 모든 디렉터리를 재귀 탐색해
    {폴더 경로(Path): json 개수(int)} 딕셔너리를 반환.
    """
    root = Path(root_dir).resolve()
    counter = Counter()

    # ① 모든 .json 경로를 재귀적으로 찾는다
    for json_path in root.rglob("*.json"):
        folder = json_path.parent            # json이 속한 디렉터리
        counter[folder] += 1

    return counter

# ────────────────────────────────
# 3.  실행
# ────────────────────────────────
def main() -> None:

    #json 데이터들이 있는 폴더
    src_dir = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\train_src').resolve()
    det_dir = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\save').resolve()

    os.makedirs(det_dir, exist_ok=True)

    all_json = count_json_per_folder(src_dir)
    total = 0
    for folder, n in all_json.items():
        rel = folder.relative_to(src_dir).as_posix() or "."   # 루트 자체는 "."
        print(f"{rel:30s} : {n}개")
        total += n
    if total == 0:
        print('변환할 파일이 없습니다.')
        exit(0)

    broken_json = defaultdict(list)
    idx = 0
    bar_len = 40
    for folder in scan_folders(src_dir):
        pid_seen.clear() 
        for json_file in iter_json_sorted(folder, natural=True):
            idx += 1           
            try:
                process_one_json(json_file, det_dir,pid_list,pid_seen)
                # 진행상황 막대그래프
                done = int(bar_len * idx / total)
                bar = '■' * done + '-' * (bar_len - done)
                print(f'진행중: [{bar}] {idx}/{total}', end='\r')
            except Exception as e:           # JSON 파싱 실패 기록
                broken_json[folder].append(json_file.name)

    print(f"완료: '{DEST_DIR}' 폴더에 변환-복사된 파일이 저장되었습니다.")

if __name__ == "__main__":
    main()
