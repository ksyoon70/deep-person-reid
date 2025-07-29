"""
Created on 2025년 5월 21일
efficientnet_vehicle_model_best_vaccxxxxx.pth 모델을 읽고, vehicle_classes.txt에서 클래스 이름을 읽어어
label_path 에서 파일을 읽어 차종을 표시하는 기능을 한다.
@author:  윤경섭
"""
import os
from pathlib import Path
import shutil
# from typing import List # List is not used from typing
from collections import defaultdict
import torch
# from torch.utils.data import DataLoader, Dataset # DataLoader, Dataset not used
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageFont, ImageDraw
import sys
import cv2
import numpy as np

# --- Configuration ---
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # Path to a .ttf font file for text overlay
DISPLAY_WINDOW_NAME = "Vehicle Recognition Result"

def display_image_with_prediction(img_pil: Image.Image, pred_class: str, confidence: float, window_name: str = DISPLAY_WINDOW_NAME, wait_key_duration: int = 0):
    """Displays an image with prediction text using OpenCV, drawing text with PIL."""
    img_cv_original = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Resize for display
    orig_height, orig_width = img_cv_original.shape[:2]
    display_width = 640
    display_height = int(orig_height * (display_width / orig_width))
    img_cv_resized = cv2.resize(img_cv_original, (display_width, display_height))

    # Prepare to draw text using PIL on the resized image
    img_pil_resized_for_text = Image.fromarray(cv2.cvtColor(img_cv_resized, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil_resized_for_text)
    
    font_size = int(display_height * 0.05) # Font size relative to resized image height
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except OSError:
        print(f"Warning: Font not found at {FONT_PATH}. Using default font.")
        font = ImageFont.load_default()
    
    label_text = f"{pred_class} ({confidence*100:.1f}%)"
    text_position = (10, 10)
    
    # Add a semi-transparent background for the text for better readability
    text_bbox = draw.textbbox(text_position, label_text, font=font)
    # Slightly pad the bounding box
    text_bg_rect = (text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5)
    
    # Create a temporary drawing surface for semi-transparent rectangle
    overlay = Image.new('RGBA', img_pil_resized_for_text.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(text_bg_rect, fill=(0, 0, 0, 128)) # Black with 50% opacity
    img_pil_resized_for_text = Image.alpha_composite(img_pil_resized_for_text.convert('RGBA'), overlay)
    
    # Draw the text on top
    final_draw = ImageDraw.Draw(img_pil_resized_for_text)
    final_draw.text(text_position, label_text, font=font, fill=(0, 255, 0, 255)) # Green text

    # Convert back to OpenCV format for display
    img_cv_display = cv2.cvtColor(np.array(img_pil_resized_for_text.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    cv2.imshow(window_name, img_cv_display)
    key = cv2.waitKey(wait_key_duration)
    return key

def main():
    # src_dir을 현재 파일이 있는 바로 위 폴더로 설정
    
    script_dir = Path(__file__).resolve().parent
    label_path = Path(r'D:/SPB_Data\deep-person-reid/vehicle_classes.txt')
    model_dir = script_dir.parent
    src_dir = Path(r'E:\윤경섭\vehicle_test')
    result_dir = Path(r'E:\윤경섭\result') # 결과 저장 폴더

    # src_dir이 없으면 생성 후 에러 표시 및 종료
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
        print(f"Error: '{src_dir.resolve()}' 폴더가 존재하지 않아 새로 생성했습니다. 이미지를 넣고 다시 실행하세요.")
        sys.exit(1)

    # result_dir 생성 (없으면)
    result_dir.mkdir(parents=True, exist_ok=True)
    print(f"결과가 저장될 폴더: {result_dir.resolve()}")

    # efficientnet_vehicle_model_best가 포함된 pth 파일 자동 탐색
    model_files = list(model_dir.glob('efficientnet_vehicle_best*.pth'))
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
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # src_dir 내 이미지 파일 추출 (하위 폴더 포함)
    image_paths = list(src_dir.rglob("*.jpg")) + list(src_dir.rglob("*.jpeg")) + list(src_dir.rglob("*.png"))
    if not image_paths:
        print(f"Error: '{src_dir}' 폴더에 이미지 파일이 없습니다.")
        sys.exit(1)

    # 이미지 그룹화
    image_groups = defaultdict(list)
    ungrouped_image_paths = []

    for img_path in image_paths:
        file_name = img_path.name
        parts = file_name.split('_')
        if len(parts) > 1 and parts[0].isdigit():
            group_id = parts[0]
            image_groups[group_id].append(img_path)
        else:
            ungrouped_image_paths.append(img_path)

    exit_processing_flag = False

    # 그룹화된 이미지 처리
    if image_groups:
        print(f"\n--- 그룹화된 이미지 처리 중 ({len(image_groups)} 그룹) ---")
    for group_id, group_paths in image_groups.items():
        if exit_processing_flag: break
        print(f"\n그룹 '{group_id}' 처리 중 ({len(group_paths)}개 이미지)...")
        
        group_predictions_info = [] # List to store (confidence, actual_pred_class, img_path)

        for img_path in group_paths:
            if exit_processing_flag: break
            try:
                img_pil = Image.open(img_path).convert('RGB')
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf_tensor, pred_tensor = torch.max(prob, 1)
                    
                    actual_confidence = conf_tensor.item()
                    pred_idx = pred_tensor.item()

                    if pred_idx not in idx_to_class:
                        print(f"  경고: {img_path.name} - 예측된 클래스 인덱스 '{pred_idx}'가 idx_to_class에 없습니다. 건너뜁니다.")
                        continue
                    actual_pred_class = idx_to_class[pred_idx]
                
                # Determine class for display and individual move (if it were ungrouped)
                display_class = "몰라" if actual_confidence <= 0.5 else actual_pred_class
                print(f"  - {img_path.name}: {display_class} ({actual_confidence*100:.1f}%)")

                # Display image
                key = display_image_with_prediction(img_pil, display_class, actual_confidence, wait_key_duration=500)
                if key == 27:  # ESC key
                    exit_processing_flag = True
                    print("ESC 키가 눌려 처리를 중단합니다.")
                    break
                
                group_predictions_info.append((actual_confidence, actual_pred_class, img_path))

            except Exception as e:
                print(f"  Error processing image {img_path.name}: {e}")
                continue
        
        if not exit_processing_flag and group_predictions_info:
            # 그룹 전체에 대한 최종 클래스 결정
            best_conf_in_group, best_actual_class_in_group, _ = max(group_predictions_info, key=lambda item: item[0])
            final_group_class_name = "몰라" if best_conf_in_group <= 0.5 else best_actual_class_in_group
            
            target_group_dir = result_dir / final_group_class_name / group_id # 그룹 ID로 하위 폴더 생성
            target_group_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  그룹 '{group_id}'의 최종 클래스: '{final_group_class_name}'. 해당 폴더로 파일 이동 중...")
            for _, _, img_path_to_move in group_predictions_info: # Move all processed images in the group
                try:
                    destination = target_group_dir / img_path_to_move.name
                    shutil.copy(str(img_path_to_move), str(destination))
                    # Copy corresponding .json file if exists
                    json_path = img_path_to_move.with_suffix('.json')
                    if json_path.exists():
                        json_destination = target_group_dir / json_path.name
                        shutil.copy(str(json_path), str(json_destination))
                    # print(f"    Copied: {img_path_to_move.name} -> {destination}")
                except Exception as e:
                    print(f"    Error moving file {img_path_to_move.name} to {destination}: {e}")

    # 그룹화되지 않은 이미지 처리
    if not exit_processing_flag and ungrouped_image_paths:
        print(f"\n--- 그룹화되지 않은 이미지 처리 중 ({len(ungrouped_image_paths)}개) ---")
    for img_path in ungrouped_image_paths:
        if exit_processing_flag: break
        try:
            img_pil = Image.open(img_path).convert('RGB')
            input_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)
                conf_tensor, pred_tensor = torch.max(prob, 1)

                actual_confidence = conf_tensor.item()
                pred_idx = pred_tensor.item()

                if pred_idx not in idx_to_class:
                    print(f"  경고: {img_path.name} - 예측된 클래스 인덱스 '{pred_idx}'가 idx_to_class에 없습니다. 건너뜁니다.")
                    continue
                actual_pred_class = idx_to_class[pred_idx]

            effective_class_name = "몰라" if actual_confidence <= 0.5 else actual_pred_class
            print(f"  - {img_path.name}: {effective_class_name} ({actual_confidence*100:.1f}%)")

            key = display_image_with_prediction(img_pil, effective_class_name, actual_confidence, wait_key_duration=500)
            if key == 27: # ESC
                exit_processing_flag = True
                print("ESC 키가 눌려 처리를 중단합니다.")
                break
            
            target_ind_dir = result_dir / effective_class_name
            target_ind_dir.mkdir(parents=True, exist_ok=True)
            destination = target_ind_dir / img_path.name
            shutil.copy(str(img_path), str(destination))
            # Copy corresponding .json file if exists
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                json_destination = target_ind_dir / json_path.name
                shutil.copy(str(json_path), str(json_destination))
            # print(f"    Copied: {img_path.name} -> {destination}")

        except Exception as e:
            print(f"  Error processing ungrouped image {img_path.name}: {e}")
            continue

    cv2.destroyAllWindows()
    print("\n모든 작업 완료.")

if __name__ == "__main__":
    main()
