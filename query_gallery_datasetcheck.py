"""
Created on Wed Sep 11 14:57:05 2024

@author: 윤경섭
본 프로그램은 해당 폴더에서 특정 폴더로 영상을 복사 한 후 비율에 따라 train, val 폴더에 복사한다.
"""

import os, shutil
import sys, random,re
from pathlib import Path, PureWindowsPath
#------------------------------------------------
#수정할 파라미터 이름
src_dir = r'D:\SPB_Data\deep-person-reid\Market-1501-v15.09.15\market1501'
#------------------------------------------------



src_dir = os.path.normpath(src_dir)
src_dir = Path(src_dir)

ROOT_DIR = os.path.dirname(__file__)

def get_subdirectories(folder_path):
    # 지정된 폴더 내의 모든 하위 항목을 가져옴
    all_items = os.listdir(folder_path)
    
    # 하위 항목 중에서 폴더만 필터링
    subdirectories = [item for item in all_items if os.path.isdir(os.path.join(folder_path, item))]
    
    return subdirectories

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if not os.path.exists(src_dir) :
    print("Error : source images folder exists. check the folder : {}".format(src_dir))
    sys.exit(0)

train_dir = os.path.join(src_dir, 'bounding_box_train')
if not os.path.exists(train_dir) :
    print("Error : source images folder exists. check the folder : {}".format(src_dir))
    sys.exit(0)

gallery_dir = os.path.join(src_dir, 'bounding_box_test')
if not os.path.exists(gallery_dir) :
    print("Error : source images folder exists. check the folder : {}".format(src_dir))
    sys.exit(0)

query_dir = os.path.join(src_dir, 'query')
if not os.path.exists(query_dir) :
    print("Error : source images folder exists. check the folder : {}".format(src_dir))
    sys.exit(0)



#각 데이터를 표시한다.
train_filenames = [filename for filename in os.listdir(train_dir) if not os.path.isdir(filename)]
gallery_filenames = [filename for filename in os.listdir(gallery_dir) if not os.path.isdir(filename)]
query_filenames = [filename for filename in os.listdir(query_dir) if not os.path.isdir(filename)]

max_id = 0

for fname in train_filenames:
    try:
        id = int(fname[0:4])
        if id > max_id:
            max_id = id
    except ValueError:
        continue

for fname in gallery_filenames:
    try:
        id = int(fname[0:4])
        if id > max_id:
            max_id = id
    except ValueError:
        continue

for fname in query_filenames:
    try:
        id = int(fname[0:4])
        if id > max_id:
            max_id = id
    except ValueError:
        continue

for id in range(1,max_id+1):
    pattern = f'{id:04}'
    tcount = sum(1 for filename in train_filenames if pattern == filename[0:4])
    gcount = sum(1 for filename in gallery_filenames if pattern == filename[0:4])
    qcount = sum(1 for filename in query_filenames if pattern == filename[0:4])
    if tcount== 0 and (gcount == 0 or qcount == 0):
        print(f'DATASET ERROR!!! ID:{id:>6}  TRAIN:{tcount:>5}   GALLERY:{gcount:>5}   QUERY:{qcount:>5}  ')
    print(f'ID:{id:>6}  TRAIN:{tcount:>5}   GALLERY:{gcount:>5}   QUERY:{qcount:>5}  ')

print('작업이 종료되었습니다.')



