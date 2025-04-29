#이 파일은 파일 이름 형식을 veri format에서 market-1501 파일 형식으로 바꾼다.
#즉 c001을 c1s1 형태로 바꾼다.

import os
import shutil
import json
from glob import glob

def convert_name(filename):
    # 예: 0002_c002_00030600_0.jpg -> 0002_c2s1_00030600_0.jpg
    parts = filename.split('_')
    if len(parts) < 4:
        return filename  # 예외처리
    cam = parts[1]
    # cam[0] == 'c'이고 cam[2] == 's'이면 업데이트하지 않음
    if not (len(cam) > 2 and cam[0] == 'c' and cam[2] == 's'):
        cam_num = cam[1:].lstrip('0')  # '002' -> '2'
        new_cam = f'c{cam_num}s1'
        parts[1] = new_cam
    # parts[2]는 6자리로 맞춤
    if len(parts[2]) != 6:
        parts[2] = parts[2].zfill(6)[-6:]
    # parts[3]는 2자리로 맞춤 ('.'으로 나누어진 앞부분이 1글자면 2글자로)
    if '.' in parts[3]:
        prefix, *rest = parts[3].split('.', 1)
        if len(prefix) == 1:
            prefix = prefix.zfill(2)
        parts[3] = '.'.join([prefix] + rest) if rest else prefix
    elif len(parts[3]) == 1:
        parts[3] = parts[3].zfill(2)
    return '_'.join(parts)

def main():
    # 현재 파일 기준 경로로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, '..', 'VeRi', 'veri', 'image_test')
    dst_dir = os.path.join(base_dir, '..', 'VeRi', 'veri', 'market1501')

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 파일 리스트 (jpg, json)
    img_files = sorted(glob(os.path.join(src_dir, '*.jpg')))
    json_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob(os.path.join(src_dir, '*.json'))}

    total = len(img_files)
    if total == 0:
        print('변환할 이미지가 없습니다.')
        exit(0)

    bar_len = 40

    for idx, img_path in enumerate(img_files, 1):
        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)
        new_name = convert_name(base)
        new_img_path = os.path.join(dst_dir, new_name)
        # 이미지 복사
        shutil.copy2(img_path, new_img_path)

        # json 처리
        if name in json_files:
            with open(json_files[name], 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['imagePath'] = new_name
            new_json_name = os.path.splitext(new_name)[0] + '.json'
            new_json_path = os.path.join(dst_dir, new_json_name)
            with open(new_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # 진행상황 막대그래프
        done = int(bar_len * idx / total)
        bar = '■' * done + '-' * (bar_len - done)
        print(f'진행중: [{bar}] {idx}/{total}', end='\r')

    print('\n변환이 모두 완료되었습니다.')

if __name__ == "__main__":
    main()