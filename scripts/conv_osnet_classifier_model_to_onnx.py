import torch
import torchreid
import torch.nn as nn
import warnings
from torchreid.utils import load_checkpoint
from collections import OrderedDict
import argparse
from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
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

def custom_load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()

    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights from "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded imagenet pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )

# OSNet 모델을 위한 래퍼 클래스
class OSNetWrapper(nn.Module):
    def __init__(self, model):
        super(OSNetWrapper, self).__init__()
        self.model = model
        # classifier 레이어의 가중치와 편향을 저장
        if hasattr(model, 'classifier'):
            self.classifier = model.classifier
    
    def forward(self, x):
        features = self.model(x)
        if hasattr(self, 'classifier'):
            classifier_output = self.classifier(features)
            return features, classifier_output
        return features





def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config-file', type=str, default='', help='path to config file'
    )
    parser.add_argument(
        '-s',
        '--sources',
        type=str,
        nargs='+',
        help='source datasets (delimited by space)'
    )
    parser.add_argument(
        '-t',
        '--targets',
        type=str,
        nargs='+',
        help='target datasets (delimited by space)'
    )
    parser.add_argument(
        '--root', type=str, default='', help='path to data root'
    )
    parser.add_argument(
        '--result', type=str, default='', help='path to result directory'
    )
    parser.add_argument(
        '--color_label', type=str, default='', help='path to color label file'
    )
    parser.add_argument(
        '--type_label', type=str, default='', help='path to type label file'
    )
    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line'
    )
    args = parser.parse_args()

    cfg = get_default_config()

    cfg.use_gpu = torch.cuda.is_available()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # leftover 인자를 cfg에 반영
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    
    model_name = 'osnet_x1_0'
    # Load the OSNet model
    model = torchreid.models.build_model(name=model_name, num_classes=594)
    #model.eval()
    #torchreid.utils.load_pretrained_weights(model, r'D:\SPB_Data\deep-person-reid\log\osnet_x1_0_veri_softmax\model\model.pth.tar-3')
    custom_load_pretrained_weights(model, cfg.model.load_weights)
    model_dict = model.state_dict()
    print(model_dict['classifier.weight'].shape)  # 또는
    print(model_dict['classifier.bias'].shape)
    # Get classifier weights and biases from the model
    if hasattr(model, 'classifier'):
        classifier_weight = model.classifier.weight.data.clone()
        classifier_bias = model.classifier.bias.data.clone()
        
        # Update model's state dict with classifier parameters
        model_dict = model.state_dict()
        model_dict['classifier.weight'] = classifier_weight
        model_dict['classifier.bias'] = classifier_bias
        model.load_state_dict(model_dict)

    # Custom Model 생성 (Global Average Pooling을 Average Pooling으로 교체하고 추가 연산 적용)
    if model_name == 'resnet18':
        model = CustomModel(model)
    else:
        # OSNet 모델을 래퍼로 감싸기
        model = OSNetWrapper(model)


    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 256, 128)

    # Export the model to ONNX
    #torch.onnx.export(model, dummy_input, 'feature.onnx', export_params=True, opset_version=11)
    # Export the model to ONNX with input and output names

    onnx_output_path = 'feature_{}.onnx'.format(model_name)
    
    torch.onnx.export(
    model,                   # 모델 객체
    dummy_input,             # 더미 입력 데이터
    onnx_output_path,        # 저장할 ONNX 파일 경로
    export_params=True,      # 학습된 파라미터를 내보내기
    opset_version=11,        # ONNX Opset 버전
    do_constant_folding=True, # 상수 폴딩 최적화
    input_names=['images'],   # 입력 텐서 이름
    output_names=['features', 'classifier_output'],  # 출력 텐서 이름
    dynamic_axes={
        'images': {0: 'batch_size'},
        'features': {0: 'batch_size'},
        'classifier_output': {0: 'batch_size'}
    }
)

if __name__ == '__main__':
    main()
