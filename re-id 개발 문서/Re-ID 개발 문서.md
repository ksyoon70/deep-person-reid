# 1. 한국 자동차 re-id 데이터 생성
## 1.1 VeRi-776 데이터 준비
한국 자동차 re-id 데이터 세트를 만들기 위해서는 [[VeRi-776]]을 기반으로 한국 차량 데이터를 추가한다.
VeRi dataset은 deep-person-reid 폴더 아래 VeRi\veri 아래에 있다.
이 아래에는 중요한 3개의 폴더가 있다.
- image_train
- image_query
- image_test
**image_train** 폴더는 OSNet으로 train을 하기 위한 파일들이 들어 있다.
**image_test** 폴더는 OSNet에서 test(gallery)하기 위한 파일들이 있다.
**image_query** 폴더는 image_test에 있는  이미지를 query를 하기 위한 파일들이 들어가 있는 폴더이다. 평가를 할때 이 파일에 있는 차량의 특징(feature)와 같은 이미지가 있를 image_test에서 검색한다. 그러므로 여기에 있는 파일은 image_test에서 임의로 뽑아서 만든 파일들이다.

Dataset의 생성은 VeRi-776 데이터에 한국 차량 데이터를 추가하는 방식으로 이루어진다. 그러므로 VeRi-776 데이터는 항상 원본으로 가지고 있어야 한다.
이때 VeRi-776 Dataset은 이미지에 해당하는 동일한 json을 가지고 있어야 한다.

## 1.2 한국 차량 데이터 준비

deep-person-reid\scripts에는 추가적이 python script가 있어 이를 사용하여 dataset을 구성한다.
예를 들어  아래와 같이 한국 자동차의 차종 폴더가 있다고 하면
![[Pasted image 20250617140149.png]]
참고로 첫번째 항목은 모델명, 두번째 항목은 상세모델명, 셋째는 색, 넷째는 차량 갯수이다. 같은 모델 같은 색이라 하더라도 광고, 적재물 등이 다르면 다른 차량으로 분류한다. 하지만, 다른 장소에서 찍혔더라도 모양과 색이 완전히 같으면 같은 차량으로 분류한다.
아래든 분류 예이다.
![[Pasted image 20250806101207.png]]

## 1.3 VeRi-776 image_query에 json 추가

**add_veri_query_set_json.py** 파을을 적용하면 query 이미지 디렉토리와 갤러리 이미지 디렉토리에 json이 없다면, 같은 파일이 gallery 디렉토리에 있는지 확인하고 해당 json을 복사한다.
이 작업은 image_query에 이미지만 있을때 적용한다.
있다면 gallery 디렉토리에 있는 json 파일을 query 디렉토리에 복사합니다.
![[Pasted image 20250805193222.png]]
복사하면 위와 같이 이미지와 json 파일이 쌍을 이루는 것을 알 수 있다.
## 1.4 VeRi-776에 한국 차량 데이터 합치기

**make_veri_korean_dataset.py** 스크립트를 통하여
VeRi-776 데이터와 한국 차량 데이터를 합친다.
main 함수 내에 아래와 같은 내용이 있다.
### --------------------------------------------------
### 사용자 기본 경로 설정 (필요 시 수정)
### --------------------------------------------------
src_dir    = Path(r'E:\윤경섭\상세차종_re-id_datasets').resolve()
vsrc_dir   = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\image_train').resolve()
vmodel_dir = Path(r'E:\윤경섭\vmodel').resolve()
det_dir    = Path(r'E:\윤경섭\save').resolve()
### color 및 type 레퍼런스 파일 경로
color_path = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\list_color.txt').resolve()
type_path = Path(r'D:\SPB_Data\deep-person-reid\VeRi\veri\list_type.txt').resolve()


### 분할 비율
TRAIN_RATIO = 0.8
TEST_RATIO  = 1.0  - TRAIN_RATIO
QUERY_RATIO = 0.15       ### test에서 폴더별로 몇 %의 데이터를 query로 쓸지 결정하는 값


src_dir 이 한국 차량데이터가 있는 폴더이다. 여기는 차량모델, 상세모델, color, 댓수가 폴더이름으로 표시된다. 
vsrc_dir은 VeRi-776의 train 폴더를 지정한다.
vmodel_dir 은 모델명_상세모델명 폴더로 만들어지는 임시 폴더이다.
det_dir 은 image_query, image_test (gallery) , image_train 으로 폴더가 만들어지고, 내부에는 이미지와 json 쌍이 저장된다.
👉 Query의 비율은 전체의 비율이 아니고, test(gallery) 파일 중 에서의 비율이다.

image_query, image_test (gallery) , image_train 이 폴더를 D:\SPB_Data\deep-person-reid\VeRi\veri 아래에 복사혀면 기존의 VeRi-776 데이터와 합쳐진다.


# 2. OSNet config 파일 
deep-person-reid\configs 폴더 아래에 보면 여러 yaml 파일들이 있다. 그중에 **[[im_osnet_x1_0_softmax_256x256_amsgrad_veri.yaml]]** 파일이 있다. 이것은 기존 파일을 조금 수정하고  이름을 변경하였다.

# 3. training
ReId OSNet은 vscode에서 training을 한다. 이를 위한 설정은 [[launch.json]] 에서 한다.
vscode에서 
![[Pasted image 20250806103200.png]] 
을 실행한다.

## 3.1 training options
[[launch.json]] 파일에서 output_usage에 따라 "feature"이면 feature vector 512차원 벡터만을 training하고, "mixture" 라고 하면 feature vector 외에 추가로 color와 vehicle type까지 training 한다.
따라서 이 값에 따라 num_classes의 갯수가 달라진다.

## 3.2 training 코드 변경
좀 더 효과적인 training을 위하여 일부 코드를 변경 하였다.
### 3.2.1 점진적 전이학습
engine.py 파일의 def __init__  함수 내에 아래 코드를 추가한다.
클래스에 학습 계획을 속성으로 추가
        self.unfreeze_schedule = {
            0: ['classifier'],
            10: ['classifier', 'conv5'],
            20: ['classifier', 'conv5', 'conv4'],
            30: ['classifier', 'conv5', 'conv4', 'conv3'],
            40: ['classifier', 'conv5', 'conv4', 'conv3', 'conv2'],
            50: ['classifier', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'],
            60 : 'all'  # 'all'로 설정하면 모든 레이어를 학습 가능하게 설정
        }
이는 각 epoch별로 어떤 layer까지 training 할지를 정하는 코드이다.
def train 함수 에서 학습할 layer를 설정하는 코드는 다음과 같다.
	if self.epoch in self.unfreeze_schedule:
        layers_to_open = self.unfreeze_schedule[self.epoch]
        print(f"Epoch {self.epoch}: Unfreezing layers: {layers_to_open}")
            
        # 1. 학습할 레이어 설정
        self.set_trainable_layers(layers_to_open)

## 3.3 training 결과
training을 시키면 아래와 같이 로그를 표시하며 훈련이 된다.
![[Pasted image 20250806111738.png]]
위 로그는 epoch 5마다 evaluation을 하며 모델을 저장하고, max epoch은 50인 상태이다.
그러면 모델은 deep-person-reid\log\osnet_x1_0_veri_softmax\model에 쌓이게 된다.

# 4. 모델을 onnx 변환
모델을 onnx로 변환하는 것은 **conv_model_to_onnx.py** 파일을 이용한다.
이 파일 내에
모델을 이름을 아래와 같이 설정한다.
model_name = 'osnet_x1_0'
모델의 output에서의 class 갯수를 설정한다.
model = torchreid.models.build_model(name=model_name, num_classes=512)
모델의 위치를 설정한다.
torchreid.utils.load_pretrained_weights(model, r'D:\SPB_Data\deep-person-reid\log\osnet_x1_0_veri_softmax\model\\model.pth.tar-50')
모델의 입력이미지 크기를 설정한다.
dummy_input = torch.randn(1, 3, 256,256)
onnx 파일 경로 및 이름을 설정한다.
onnx_output_path = 'feature.onnx'

이를 설정하고 수행을 하면 onnx 파일이 생성된다.

# 5. yolo v8 onnx batch 변환
yolo v8 onnx 파일로 배치 변환을 하려면 **export_yolo2onnx_with_batch_efficientNMS.py** 파일을 사용한다.
이 파일은
YOLOv8 PyTorch 모델(model.pt)을 TensorRT 추론에 최적화된 ONNX 모델로 변환하는 자동화 파이프라인입니다.

스크립트의 전체적인 기능은 원본 PyTorch 모델에 동적 배치 크기를 적용하고, 모델을 경량화한 뒤, TensorRT의 고성능 NMS(Non-Maximum Suppression) 플러그인을 그래프에 직접 삽입하여 후처리까지 포함된 최종 ONNX 파일을 만드는 것입니다.

코드는 아래와 같다.

import os
import torch
from ultralytics import YOLO
import onnx
from onnxsim import simplify
import onnx_graphsurgeon as gs
import numpy as np

### --- 1. 설정 ---
PT_FILE = "model.pt"  # 입력할 YOLOv8 Pytorch 모델 파일
TEMP_ONNX = "temp_model.onnx" # 중간 과정에서 생성될 임시 ONNX 파일
SIMPLE_ONNX = "model_simplified.onnx"
FINAL_ONNX = "model_with_nms.onnx" # 최종 결과물

### NMS 플러그인 속성
MAX_OUTPUT_BOXES = 100
SCORE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
NUM_CLASSES = 4  # 사용하는 모델의 클래스 수에 맞게 조정 (예: COCO는 80)

### --- 2. .pt 모델 로드 및 임시 ONNX 파일로 변환 ---
print(f"'{PT_FILE}' 파일을 로드하여 ONNX로 변환합니다...")
model = YOLO(PT_FILE)

# torch.onnx.export는 순수 nn.Module이 필요하므로 내부 PyTorch 모델을 가져옵니다.
pytorch_model = model.model.eval()

### --- 3. ONNX 변환을 위한 더미 입력(dummy input) 생성 ---
# 입력 형태에 맞춰 더미 입력을 생성합니다. (배치크기=1, 채널=3, 높이=640, 너비=640)
### 이 입력은 모델 구조를 추적(trace)하는 데 사용됩니다.
dummy_input = torch.randn(1, 3, 640, 640)

### --- 4. torch.onnx.export를 사용하여 ONNX로 변환 ---
### dynamic_axes를 설정하여 0번 축(배치)을 가변 크기로 만듭니다.
print("PyTorch 모델을 ONNX로 변환 중...")
torch.onnx.export(
    pytorch_model,
    dummy_input,
    TEMP_ONNX,
    input_names=['images'],
    output_names=['output0'], # YOLOv8의 출력은 보통 1개이므로 output0으로 지정
    dynamic_axes={
        'images': {0: 'batch_size'},  # 입력의 0번 축 (배치)
        'output0': {0: 'batch_size'} # 출력의 0번 축 (배치)
    },
    opset_version=12
)
print(f"ONNX 모델 저장 완료: {TEMP_ONNX}")

print("ONNX 모델을 최적화(simplify)하는 중...")
onnx_model = onnx.load(TEMP_ONNX)
model_simplified, check = simplify(onnx_model)
assert check, "ONNX 모델 최적화에 실패했습니다."
onnx.save(model_simplified, SIMPLE_ONNX)
print(f"최적화된 ONNX 모델 저장 완료: {SIMPLE_ONNX}")

### --- 5. ONNX GraphSurgeon으로 NMS 플러그인 추가 ---

### int64 타입의 ONNX Constant 노드를 생성하는 헬퍼 함수
def C(name, vals):
    return gs.Constant(name=name, values=np.asarray(vals, dtype=np.int64))

print(f"'{TEMP_ONNX}' 파일을 로드하여 GraphSurgeon 작업을 시작합니다...")
graph = gs.import_onnx(onnx.load(SIMPLE_ONNX))

### 입력/출력 노드 가져오기
inp = graph.inputs[0]
raw_output = graph.outputs[0]
batch_dim_symbol = inp.shape[0] # ultralytics가 지정한 동적 배치 심볼 (예: 'batch')

### 모델의 원시 출력을 NMS 입력에 맞게 전처리 (Transpose, Slice)
transposed_output = gs.Variable(name="transposed_predictions", dtype=np.float32)
graph.nodes.append(gs.Node(op="Transpose", inputs=[raw_output], outputs=[transposed_output], attrs={"perm":[0, 2, 1]}))

boxes = gs.Variable(name="boxes", dtype=np.float32)
scores = gs.Variable(name="scores", dtype=np.float32)
graph.nodes.append(gs.Node(
    op="Slice",
    inputs=[transposed_output, C("slice_boxes_starts", [0]), C("slice_boxes_ends", [4]), C("slice_boxes_axes", [2])],
    outputs=[boxes]
))
graph.nodes.append(gs.Node(
    op="Slice",
    inputs=[transposed_output, C("slice_scores_starts", [4]), C("slice_scores_ends", [4 + NUM_CLASSES]), C("slice_scores_axes", [2])],
    outputs=[scores]
))
print("모델의 원시 출력을 boxes와 scores로 분리했습니다.")

### EfficientNMS_TRT 플러그인 노드 생성 및 연결
num_dets    = gs.Variable("num_dets",    dtype=np.int32,   shape=[batch_dim_symbol, 1])
nms_boxes   = gs.Variable("nms_boxes",   dtype=np.float32, shape=[batch_dim_symbol, MAX_OUTPUT_BOXES, 4])
nms_scores  = gs.Variable("nms_scores",  dtype=np.float32, shape=[batch_dim_symbol, MAX_OUTPUT_BOXES])
nms_classes = gs.Variable("nms_classes", dtype=np.int32,   shape=[batch_dim_symbol, MAX_OUTPUT_BOXES])

nms_node = gs.Node(
    op="EfficientNMS_TRT",
    name="EfficientNMS",
    domain="ai.onnx.contrib",
    inputs=[boxes, scores],
    outputs=[num_dets, nms_boxes, nms_scores, nms_classes],
    attrs={
        "plugin_version": "1", "background_class": -1, "max_output_boxes": MAX_OUTPUT_BOXES,
        "score_threshold": SCORE_THRESHOLD, "iou_threshold": IOU_THRESHOLD, "box_coding": 1,
        "score_activation": False, "class_agnostic": False,
    }
)
graph.nodes.append(nms_node)
print("EfficientNMS_TRT 플러그인 노드를 그래프에 추가했습니다.")

### 그래프의 최종 출력을 NMS 출력으로 교체하고 정리
graph.outputs = [num_dets, nms_boxes, nms_scores, nms_classes]
graph.cleanup().toposort()

### --- 4. 최종 ONNX 파일 저장 및 임시 파일 삭제 ---
onnx_model = gs.export_onnx(graph)
onnx.save(onnx_model, FINAL_ONNX)
print(f"✅ 작업 완료! 최종 모델이 '{FINAL_ONNX}' 파일로 저장되었습니다.")

### 임시 파일 삭제
os.remove(TEMP_ONNX)
print(f"임시 파일 '{TEMP_ONNX}'을 삭제했습니다.")

os.remove(SIMPLE_ONNX)
print(f"중간과정에서 최적화된 임시 ONNX 파일 '{SIMPLE_ONNX}'을 삭제했습니다.")

### 핵심 기능 요약

YOLOv8 모델 로드 및 ONNX 변환:

ultralytics 라이브러리를 사용해 .pt 파일을 로드합니다.

 torch.onnx.export를 이용해 모델을 표준 ONNX 형식으로 변환하며, 이때  배치 크기 축을 동적(batch_size) 으로 설정하여 다양한 배치 크기로 추론할 수 있게 만듭니다.

ONNX 모델 최적화 (simplify):

onnx-simplifier 라이브러리를 사용해 변환된 ONNX 모델에서 불필요한 연산이나 중복 노드를 제거합니다.

이 과정은 모델의 파일 크기를 줄이고 추론 속도를 향상시킵니다.

EfficientNMS 플러그인 삽입:

  onnx-graphsurgeon이라는 강력한 도구를 사용해 최적화된 ONNX 모델의 내부 그래프 구조를 직접 수정합니다.

 모델의 최종 출력단에 TensorRT 전용 EfficientNMS_TRT 플러그인 노드를 추가합니다. 이를 통해 스코어 필터링, IOU(Intersection over Union) 계산, 박스 선택 등 복잡한 후처리 과정이 모델 내부에 포함됩니다.

 최종 파일 생성 및 정리:

 모든 수정이 완료된 그래프를 최종 ONNX 파일(model_with_nms.onnx)로 저장합니다.

 변환 과정에서 생성된 모든 임시 파일들을 자동으로 삭제하여 최종 결과물만 남깁니다.

결론적으로, 이 스크립트는 TensorRT 환경에서 YOLOv8 모델을 최고 성능으로 실행하기 위한 맞춤형 ONNX 파일을 한 번에 생성해주는 매우 유용한 도구이다.









