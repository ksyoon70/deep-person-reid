"""
Created on 2025년 5월 21일
efficientnet_vehicle_model_best_vaccxxxxx.pth 모델을 읽고, vehicle_classes.txt에서 클래스 이름을 읽어어
label_path 에서 파일을 읽어 차종을 표시하는 기능을 한다.
@author:  윤경섭
"""
import os
from pathlib import Path
import shutil
from typing import List
from vehicle_model_dataset_prepare import run_vmodel_dataset_prepare
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageFont, ImageDraw
import sys
import glob
import cv2
import numpy as np

DATA_PREPARE = False
def main():
    # src_dir을 현재 파일이 있는 바로 위 폴더로 설정
    
    script_dir = Path(__file__).resolve().parent
    label_path = Path(r'D:/SPB_Data\deep-person-reid/vehicle_classes.txt')
    model_dir = script_dir.parent
    src_dir = Path(r'E:\윤경섭\vehicle_test')

    # src_dir이 없으면 생성 후 에러 표시 및 종료
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        print(f"Error: '{src_dir.resolve()}' 폴더가 존재하지 않아 새로 생성했습니다. 이미지를 넣고 다시 실행하세요.")
        sys.exit(1)

    # efficientnet_vehicle_model_best가 포함된 pth 파일 자동 탐색
    model_files = list(model_dir.glob('efficientnet_vehicle_model_best*.pth'))
    if not model_files:
        print("Error: efficientnet_vehicle_model_best로 시작하는 .pth 파일이 scripts 폴더에 없습니다.")
        sys.exit(1)
    model_path = model_files[0]
    print(f"모델 파일: {model_path}")

    # 클래스 매핑 복원 (label_path가 있으면 파일에서, 없으면 폴더명 기준)
    class_to_idx = {}
    if label_path.exists():
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                labels = [line.strip() for line in f if line.strip()]
        except UnicodeDecodeError:
            with open(label_path, 'r', encoding='cp949') as f:
                labels = [line.strip() for line in f if line.strip()]
        for idx, class_name in enumerate(labels):
            class_to_idx[class_name] = idx
        print(f"레이블 파일에서 {len(class_to_idx)}개 클래스 로드")
    else:
        for idx, class_name in enumerate(sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])):
            class_to_idx[class_name] = idx
        print(f"폴더명 기준 {len(class_to_idx)}개 클래스 로드")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)
    if num_classes == 0:
        print(f"Error: '{src_dir.resolve()}' 폴더에 클래스별 하위 폴더가 없습니다.")
        sys.exit(1)

    # 모델 및 transform 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # src_dir 내 이미지 파일 추출 (하위 폴더 포함)
    image_paths = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.jpeg")) + list(src_dir.rglob("*.png"))
    if not image_paths:
        print(f"Error: '{src_dir}' 폴더에 이미지 파일이 없습니다.")
        sys.exit(1)

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
            pred_class = idx_to_class[pred.item()]
            confidence = conf.item()
        # OpenCV로 영상 출력 및 정보 표시
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # PIL 이미지로 변환
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        # 다시 OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        orig_height, orig_width = img_cv.shape[:2]
        new_width = 640
        new_height = int(orig_height * (new_width / orig_width))
        img_cv = cv2.resize(img_cv, (new_width, new_height))

        # 확대된 이미지에 텍스트 쓰기 (PIL)
        img_pil_resized = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil_resized)
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_size = int(new_height * 0.05)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
        label_text = f"{pred_class} ({confidence*100:.1f}%)"
        draw.text((10, 10), label_text, font=font, fill=(0,255,0))
        # 다시 OpenCV 이미지로 변환
        img_cv = cv2.cvtColor(np.array(img_pil_resized), cv2.COLOR_RGB2BGR)
        cv2.imshow('Result', img_cv)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

