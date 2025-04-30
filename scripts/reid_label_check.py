#이 프로그램은 src_dir에 labelme로 라벨링한 데이트의 경로를 입력하고 이상 유무를 확인하는 코드이다.
import os
import json
import shutil
from typing import Set
from glob import glob

# ────────────────────────────────
# 1.  색·종류 레퍼런스 읽기
# ────────────────────────────────
def load_first_tokens(txt_path: str) -> Set[str]:
    """각 라인의 첫 번째 토큰(token[TAB]id 형식)을 집합으로 반환"""
    tokens: Set[str] = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens.add(line.split()[0])
    return tokens


def main() -> None:
    # (1) 라벨 파일들이 있는 최상위 폴더
    src_dir = os.path.normpath(
        r'Z:\영상라벨링\작업완료\147차_시즌3(0421)\객체추적\오정은\006(0.5일)'
    )

    # (2) VeRi 기준 색·차종 레퍼런스
    base_dir   = os.path.join("VeRi", "veri")
    color_set  = load_first_tokens(os.path.join(base_dir, "list_color.txt"))
    type_set   = load_first_tokens(os.path.join(base_dir, "list_type.txt"))

    # (3) LABEL_LIST (차종_색상 + window)
    label_list = [f"{vt}_{cl}"  for vt in sorted(type_set) for cl in sorted(color_set)]
    if "window" not in label_list:
        label_list.append("window")

    # (4) 오류 파일 보관 폴더
    error_dir = os.path.join(src_dir, "label_error")

    # (5) src_dir 아래 모든 *.json 수집
    json_pattern = os.path.join(src_dir, "**", "*.json")
    json_paths   = glob(json_pattern, recursive=True)

    total, moved = len(json_paths), 0
    for jp in json_paths:
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[경고] JSON 파싱 실패: {jp} ({e})")
            continue

        # shapes[*]['label'] 포함 모든 라벨 모음
        labels = []
        if isinstance(data.get("label"), str):
            labels.append(data["label"])
        if isinstance(data.get("shapes"), list):
            labels.extend([s.get("label", "") for s in data["shapes"]])

        # LABEL_LIST에 없는 라벨이 하나라도 있으면 오류 처리
        if any(lb not in label_list for lb in labels):
            if not os.path.isdir(error_dir):          # error_dir 없으면 생성
                os.makedirs(error_dir, exist_ok=True)

            # (a) JSON 복사
            shutil.copy2(jp, os.path.join(error_dir, os.path.basename(jp)))

            # (b) 이미지 복사
            img_name = data.get("imagePath", "")
            if img_name:
                img_path = os.path.join(os.path.dirname(jp), img_name)
                if os.path.exists(img_path):
                    shutil.copy2(img_path, os.path.join(error_dir, os.path.basename(img_path)))

            moved += 1

    print(f"총 {total}개 중 {moved}개가 'label_error' 폴더로 복사되었습니다.")
    print("작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
