"""
Created on 2025년 5월 21일
한마디로 데이터 셋을 준비하는 코드이다.
src_dir 변수에 예를들어 'E:\윤경섭\상세차종_re-id_datasets' 가 할당되어 있고 
dst_dir 변수에 예를 들어 'E:\윤경섭\vehicle_model' 라고 할당 되어 있다고 하자.
1. 이 코드는 src_dir의 하위 디렉토리를 검사하여 디렉토리 이름을 '_' 로 나누어 첫번째 파트(모델이름)과 두번째 파트(상세모델)로 새로운 디렉토리를 만들어 dst_dir 폴더 아래에 하위 폴더를 만든다.
그리고 이 폴더에 src_dir 아래의 폴더 내용을 각각 dst_dir로 복사한다.
2. 각 dst_dir 폴더의 이름이 class이름이 되어 train_ratio 만큼은 train 하위 폴더에 나머지는 validation 폴더에 저장이된다.
3. model을 만들어 해당 차종이 나오도록 훈련을 시킨다.
@author:  윤경섭
"""
import os
from pathlib import Path
import shutil
from typing import List
import random

train_ratio = 0.8

def read_directories(path) -> List[str]:
    dir_names = []  # 디렉토리 이름을 저장할 리스트
    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            dir_names.append(dir_name)  # 디렉토리 이름을 리스트에 추가

    return dir_names  # 디렉토리 이름 리스트를 리턴

def create_directories(dir_names, base_path) -> (list, list):
    train_base = os.path.join(base_path, 'train')
    valid_base = os.path.join(base_path, 'valid')
    ntrain_dirs = set()
    nvalid_dirs = set()
    for name in dir_names:
        parts = name.split('_')
        if len(parts) >= 2:
            new_dir_name = '_'.join(parts[:2])
            train_dir_path = os.path.join(train_base, new_dir_name)
            valid_dir_path = os.path.join(valid_base, new_dir_name)
            os.makedirs(train_dir_path, exist_ok=True)
            os.makedirs(valid_dir_path, exist_ok=True)
            ntrain_dirs.add(train_dir_path)
            nvalid_dirs.add(valid_dir_path)
    return list(ntrain_dirs), list(nvalid_dirs)

def split_and_copy_files(src_dir, ntrain_dirs, nvalid_dirs, train_ratio=0.8):
    # For each src_dir subfolder, split files and copy to train/valid dirs
    total_pairs = 0
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            src_subdir = os.path.join(root, dir_name)
            mo_name_folder = os.path.basename(src_subdir)
            for train_dir in ntrain_dirs:
                nmo_name_folder = os.path.basename(train_dir)
                if nmo_name_folder in mo_name_folder:
                    all_files = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f))]
                    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_exts]
                    for img in image_files:
                        base = os.path.splitext(img)[0]
                        json_file = base + '.json'
                        if json_file in all_files:
                            total_pairs += 1
    if total_pairs == 0:
        print('No image/json pairs found to copy.')
        return
    copied_pairs = 0
    bar_length = 40
    def print_progress(copied, total):
        percent = copied / total
        filled = int(bar_length * percent)
        bar = '■' * filled + '-' * (bar_length - filled)
        print(f'\rProgress: [{bar}] {copied}/{total} pairs', end='')
    for root, dirs, files in os.walk(src_dir):
        for dir_name in dirs:
            src_subdir = os.path.join(root, dir_name)
            mo_name_folder = os.path.basename(src_subdir)
            for train_dir in ntrain_dirs:
                nmo_name_folder = os.path.basename(train_dir)  #모델이름만 추출한다.
                mo_name_folder_parts = mo_name_folder.split('_')
                if len(mo_name_folder_parts) >= 2:
                    model_name = '_'.join(mo_name_folder_parts[:2])
                else:
                    model_name = mo_name_folder # '_'가 없거나 하나뿐인 경우 전체 문자열 반환
                if nmo_name_folder == model_name:
                    all_files = [f for f in os.listdir(src_subdir) if os.path.isfile(os.path.join(src_subdir, f))]
                    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
                    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in image_exts]
                    pairs = []
                    for img in image_files:
                        base = os.path.splitext(img)[0]
                        json_file = base + '.json'
                        if json_file in all_files:
                            pairs.append((img, json_file))
                    random.shuffle(pairs)
                    n_total = len(pairs)
                    n_train = int(n_total * train_ratio)
                    n_valid = n_total - n_train
                    if n_valid == 0 and n_total > 0:
                        n_train = n_total - 1
                        n_valid = 1
                    train_pairs = pairs[:n_train]
                    valid_pairs = pairs[n_train:]
                    for img, js in train_pairs:
                        shutil.copy2(os.path.join(src_subdir, img), train_dir)
                        shutil.copy2(os.path.join(src_subdir, js), train_dir)
                        copied_pairs += 1
                        print_progress(copied_pairs, total_pairs)
                    valid_dir = [
                        d for d in nvalid_dirs
                        if nmo_name_folder == Path(d).name
                    ]
                    if len(valid_dir):
                        for img, js in valid_pairs:
                            shutil.copy2(os.path.join(src_subdir, img), valid_dir[0])
                            shutil.copy2(os.path.join(src_subdir, js), valid_dir[0])
                            copied_pairs += 1
                            print_progress(copied_pairs, total_pairs)
                    else:
                        print(f'train {train_dir}에 해당하는 validation 폴더가 없습니다.')
    print()

def run_vmodel_dataset_prepare(src_dir, dst_dir, train_ratio=0.8):
    src_dir = Path(src_dir).resolve()
    dst_dir = Path(dst_dir).resolve()
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    else:
        shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

    dir_names = read_directories(src_dir)
    ntrain_dirs, nvalid_dirs = create_directories(dir_names, dst_dir)
    split_and_copy_files(src_dir, ntrain_dirs, nvalid_dirs, train_ratio)

def main():
    src_dir = r'E:\윤경섭\상세차종_re-id_datasets'
    dst_dir = r'E:\윤경섭\vehicle_model'
    run_vmodel_dataset_prepare(src_dir, dst_dir, train_ratio)

if __name__ == "__main__":
    main()

