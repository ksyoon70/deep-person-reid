import os
import shutil
import random
import json
import sys
import cv2
import numpy as np
from collections import defaultdict

# 1. 경로 및 설정
src_root = r"E:\SPB_Data\chardet\datasets\vr_images"
dst_root = r"D:\SPB_Data\deep-person-reid\Ocr\vReg"
categories_path = r"E:\SPB_Data\chardet\vregion_categories.txt"
labels_path = r"E:\SPB_Data\chardet\LPR_Total_Labels.txt"

TRAIN_RATIO = 0.7
QUERY_RATIO = 0.15

def load_region_mapping(cat_path, lab_path):
    with open(cat_path, 'r', encoding='utf-8') as f:
        target_keys = {line.strip() for line in f if line.strip()}

    mapping = {}
    current_id = 1
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line: continue
            key, hangul = line.split(',')
            if key in target_keys:
                mapping[hangul] = f"{current_id:04d}"
                current_id += 1
    if not mapping:
        print("매핑된 지역 정보가 없습니다.")
        sys.exit()
    return mapping

def clear_folder_contents(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'삭제 실패: {file_path}. 사유: {e}')

def prepare_directory():
    sub_folders = ['image_train', 'image_test', 'image_query']
    existing_files = False
    for f in sub_folders:
        path = os.path.join(dst_root, f)
        if os.path.exists(path) and os.listdir(path):
            existing_files = True
            break
    if existing_files:
        res = input(f"기존 데이터가 존재합니다. 파일만 삭제할까요? (y/n): ")
        if res.lower() == 'y':
            for f in sub_folders:
                clear_folder_contents(os.path.join(dst_root, f))
        else:
            sys.exit()
    else:
        for f in sub_folders:
            os.makedirs(os.path.join(dst_root, f), exist_ok=True)

def update_json_and_copy(src_jpg, src_json, dst_dir, new_name_base):
    new_jpg_name = new_name_base + ".jpg"
    new_json_name = new_name_base + ".json"
    dst_jpg_path = os.path.join(dst_dir, new_jpg_name)
    dst_json_path = os.path.join(dst_dir, new_json_name)

    shutil.copy(src_jpg, dst_jpg_path)
    if os.path.exists(src_json):
        try:
            with open(src_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['imagePath'] = new_jpg_name
            with open(dst_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except: pass
    return dst_jpg_path, dst_json_path

def main():
    region_to_id = load_region_mapping(categories_path, labels_path)
    prepare_directory()

    data_collector = defaultdict(list)
    for root, dirs, files in os.walk(src_root):
        for filename in files:
            if filename.endswith(".jpg"):
                region_name = filename.split('_')[-1].replace('.jpg', '')
                if region_name in region_to_id:
                    src_jpg = os.path.join(root, filename)
                    src_json = src_jpg.replace(".jpg", ".json")
                    data_collector[region_name].append((src_jpg, src_json))

    for region_name, items in data_collector.items():
        class_id = region_to_id[region_name]
        random.shuffle(items)
        
        split_idx = int(len(items) * TRAIN_RATIO)
        train_items = items[:split_idx]
        test_items = items[split_idx:]
        
        # 1. Train 복사 (인덱스 1부터 시작)
        for idx, (jpg, js) in enumerate(train_items, start=1):
            new_base = f"{class_id}_c1s1_{idx:06d}_01"
            update_json_and_copy(jpg, js, os.path.join(dst_root, 'image_train'), new_base)
            
        # 2. Test 복사 (인덱스 1부터 시작)
        copied_test_files = []
        for idx, (jpg, js) in enumerate(test_items, start=1):
            new_base = f"{class_id}_c1s1_{idx:06d}_02"
            dst_p, dst_j = update_json_and_copy(jpg, js, os.path.join(dst_root, 'image_test'), new_base)
            copied_test_files.append((dst_p, dst_j))
            
        # 3. Query 샘플링 (Test에 생성된 파일 그대로 복사)
        if copied_test_files:
            query_count = max(1, int(len(copied_test_files) * QUERY_RATIO))
            query_samples = random.sample(copied_test_files, query_count)
            for q_jpg, q_json in query_samples:
                shutil.copy(q_jpg, os.path.join(dst_root, 'image_query', os.path.basename(q_jpg)))
                if os.path.exists(q_json):
                    shutil.copy(q_json, os.path.join(dst_root, 'image_query', os.path.basename(q_json)))

        print(f"완료: {region_name}({class_id}) | Train: {len(train_items)} | Test: {len(test_items)}")

    print(f"\n작업이 완료되었습니다.")

if __name__ == "__main__":
    main()