import torch
import torchreid
import torch.nn as nn


# Custom Model: ResNet18에서 Global Average Pooling을 Average Pooling으로 교체하고 추가 연산 수행
class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        # layer4 이후 전체를 사용하고, Global Average Pooling을 Average Pooling으로 대체
        self.avgpool = nn.AvgPool2d(kernel_size=(4, 2),stride=1)  # 새로운 Average Pooling 설정

    def forward(self, x):
        # ResNet18의 기본 레이어 통과
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)  # layer4 전체를 사용

        # Average Pooling 적용
        x = self.avgpool(x)
        
        # Average Pooling 이후 추가 연산 단계
        # 1. Reshape (Flatten)
        x = x.view(x.size(0), -1)  # 피처 맵을 벡터로 변환 (N, 512)
        x_out = x
        
        # 2. 절댓값 연산
        x = torch.abs(x)
        
        # 3. 각 요소에 제곱 연산
        x_pow = torch.pow(x, 2)
        
        # 4. 제곱합 계산
        sum_pow = torch.sum(x_pow, dim=1, keepdim=True)
        
        # 5. 제곱근 계산 (루트)
        sqrt_sum = torch.pow(sum_pow, 0.5)
        
        # 6. 나눗셈 (L2 정규화)
        output = x_out / sqrt_sum
        
        return output

model_name = 'resnet18'
# Load the OSNet model
model = torchreid.models.build_model(name=model_name, num_classes=1000)
torchreid.utils.load_pretrained_weights(model, r'D:\SPB_Data\deep-person-reid\log\resnet18_128x64__market1501_softmax\model\model.pth.tar-40')


# Custom Model 생성 (Global Average Pooling을 Average Pooling으로 교체하고 추가 연산 적용)
if model_name == 'resnet18':
    model = CustomModel(model)

# Set the model to evaluation mode
model.eval()

# Create dummy input tensor
dummy_input = torch.randn(1, 3, 128,64)

# Export the model to ONNX
#torch.onnx.export(model, dummy_input, 'feature.onnx', export_params=True, opset_version=11)
# Export the model to ONNX with input and output names

onnx_output_path = 'feature.onnx'
torch.onnx.export(
    model,                   # 모델 객체
    dummy_input,             # 더미 입력 데이터
    onnx_output_path,        # 저장할 ONNX 파일 경로
    export_params=True,      # 학습된 파라미터를 내보내기
    opset_version=11,        # ONNX Opset 버전
    do_constant_folding=True, # 상수 폴딩 최적화
    input_names=['images'],   # 입력 텐서 이름
    output_names=['output0'], # 출력 텐서 이름
    dynamic_axes={'images': {0: 'batch_size'}, 'output0': {0: 'batch_size'}} # 배치 크기 동적 지원
)