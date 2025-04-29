# infer2.py는 rank1 출력을 하기 보다는 test 데이터를 읽어서 
# pid가 무엇인지, 그리고 색과 모델이 무엇인지를 출력하는 것을 목표로 함.
# 작성자 : 윤경섭. 2025.04.16
import sys
import time
import os
import os.path as osp
import argparse
import torch
import torch.nn as nn
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torchreid
from torchreid.utils import load_checkpoint
from collections import OrderedDict
from torchreid.data.vehicle_datamanager import VehicleImageDataManager
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)

from torchreid import metrics
from torchreid.utils import (
    MetricMeter, AverageMeter, re_ranking, open_all_layers, save_checkpoint,
    open_specified_layers, visualize_ranked_results
)

from torch.nn import functional as F
import shutil


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        if cfg.data.root == 'VeRi':  # VeRi 데이터셋인 경우
            return torchreid.data.VehicleImageDataManager(**imagedata_kwargs(cfg))
        else:
            return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


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

def load_model(cfg, datamanager):
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=(datamanager.num_train_pids + datamanager.num_train_color_ids + datamanager.num_train_type_ids) if cfg.data.root == 'VeRi' else datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    

    return model
    


def parse_data_for_eval(data, cfg):
    imgs = data['img']
    pids = data['pid']
    camids = data['camid']
    impaths = data['impath']  # Add impath
    if cfg.data.root == 'VeRi':
        colors = data['color_id']
        typeids = data['type_id']
        return imgs, pids, camids, colors, typeids, impaths
    else:
        return imgs, pids, camids, None, None, impaths


def load_label_file(file_path):
    with open(file_path, 'r') as f:
        # Split each line by space and take only the label text (after the number)
        labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    return labels

def load_label_map(file_path):
    """
    list_color.txt 파일을 읽어서
    숫자(int)를 key, 색 이름(str)을 value로 하는 dict 반환.
    """
    color_map = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 공백으로 분리. 마지막 토큰은 숫자, 나머지는 색 이름
            parts = line.split()
            *name_parts, num_str = parts
            name = "".join(name_parts)    # 예: ['redorange'] -> "redorange"
            try:
                num = int(num_str) - 1
            except ValueError:
                # 숫자로 변환할 수 없으면 건너뜀
                continue
            color_map[num] = name
        color_map[-1] = 'uk'
    return color_map


def inference(cfg, model, datamanager, label_files, normalize_feature=False, dist_metric='euclidean', use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False):
    # Load label files

    use_xml_file = False
    if use_xml_file == True:
        color_labels = load_label_file(label_files['color'])
        type_labels = load_label_file(label_files['type'])
    else:
        color_labels = load_label_map(label_files['color'])
        type_labels = load_label_map(label_files['type'])
    
    # Create result directory if it doesn't exist
    os.makedirs(cfg.result, exist_ok=True)

    # Get query and gallery loaders
    targets = list(datamanager.test_loader.keys())

    for name in targets:
        domain = 'source' if name in datamanager.sources else 'target'
        print('##### Evaluating {} ({}) #####'.format(name, domain))
        #query_loader = datamanager.test_loader[name]['query']
        gallery_loader = datamanager.test_loader[name]['gallery']
        break

    batch_time = AverageMeter()

    def _feature_extraction(data_loader, model):
            f_, pids_, o_pids_, camids_, colors_, o_colors_, typeids_, o_typeids_, impaths_ = [], [], [], [], [], [], [], [], []
            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader):
                    imgs, pids, camids, colors, typeids, impaths = parse_data_for_eval(data, cfg)
                    """
                    # interactive 모드 켜기 (윈도우 하나만 띄워놓고 갱신)
                    plt.ion()
                    # 1) 한 번만 Figure와 Axes 생성
                    fig, ax = plt.subplots()
                    arr = np.array(imgs[0])
                    arr_hwc = np.transpose(arr, (1, 2, 0))  # (H, W, C)로 변경
                    # 2) 이전 이미지를 지우고
                    ax.clear()
                    # 3) 새로운 이미지를 그리기
                    ax.imshow(arr_hwc)
                    ax.set_title(f"Frame {data['impath'][0]}")
                    # 4) 화면 갱신
                    fig.canvas.draw()
                    plt.pause(0.1)       # 잠깐 멈춰줘야 갱신이 화면에 반영됨
                    """
                    if cfg.use_gpu:
                        imgs = imgs.cuda()
                    end = time.time()
                    features = model(imgs)
                    batch_time.update(time.time() - end)
                    features = features.cpu()
                    outputs = features
                    if cfg.data.root == 'VeRi':
                        pid_end = datamanager.num_train_pids                 # 575
                        color_end = pid_end + datamanager.num_train_color_ids # 575 + 10 = 585
                        logits_pid   = outputs[:, :pid_end]        # shape (B, 575)
                        logits_color = outputs[:, pid_end :color_end]     # shape (B, 10)
                        logits_type  = outputs[:, color_end:]        # shape (B, 9)
                        # Get predicted labels using argmax
                        pred_pids = torch.argmax(logits_pid, dim=1)
                        pred_colors = torch.argmax(logits_color, dim=1)
                        pred_types = torch.argmax(logits_type, dim=1)
                        output = {
                            'pid': pred_pids,
                            'color': pred_colors,
                            'type': pred_types
                        }
                    f_.append(features)
                    pids_.extend(pids.tolist())
                    o_pids_.extend(output['pid'].tolist())
                    camids_.extend(camids.tolist())
                    colors_.extend(colors.tolist())
                    o_colors_.extend(output['color'].tolist())
                    typeids_.extend(typeids.tolist())
                    o_typeids_.extend(output['type'].tolist())
                    impaths_.extend(impaths)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            o_pids_ = np.asarray(o_pids_)
            camids_ = np.asarray(camids_)
            colors_ = np.asarray(colors_)
            o_colors_ = np.asarray(o_colors_)
            typeids_ = np.asarray(typeids_)
            o_typeids_ = np.asarray(o_typeids_)
            impaths_ = np.asarray(impaths_)
            return f_, pids_, o_pids_, camids_, colors_, o_colors_, typeids_, o_typeids_, impaths_
    
    #print('Extracting features from query set ...')
    #qf, q_pids, q_camids, q_colors, q_o_colors, q_typeids, q_o_typeids, q_impaths = _feature_extraction(query_loader, model)
    #print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    print('Extracting features from gallery set ...')
    gf, g_pids, g_o_pids, g_camids, g_colors, g_o_colors, g_typeids, g_o_typeids, g_impaths = _feature_extraction(gallery_loader, model)
    print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('Speed: {:.4f} sec/batch'.format(batch_time.avg))
    """
    if normalize_feature:
        print('Normalzing features with L2 norm ...')
        qf = F.normalize(qf, p=2, dim=1)
        gf = F.normalize(gf, p=2, dim=1)

    print('Computing distance matrix with metric={} ...'.format(dist_metric))
    distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    distmat = distmat.numpy()

    if rerank:
        print('Applying person re-ranking ...')
        distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
        distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
        distmat = re_ranking(distmat, distmat_qq, distmat_gg)
    
    int('Computing CMC and mAP ...')
    cmc, mAP = metrics.evaluate_rank(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_metric_cuhk03=use_metric_cuhk03
    )

    print('** Results **')
    print('mAP: {:.1%}'.format(mAP))
    print('CMC curve')
    for r in ranks:
        print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
    """
    total_images = len(g_impaths)
    acc_pid = 0
    acc_color = 0
    acc_type = 0
    # Save results with labels
    for i, (g_impath,g_pid ,g_o_pid, g_color, g_o_color, g_type, g_o_type) in enumerate(zip(g_impaths,g_pids, g_o_pids ,g_colors, g_o_colors, g_typeids, g_o_typeids)):
        # Get filename without path and extension
        filename = os.path.splitext(os.path.basename(g_impath))[0]
        # Get color and type labels
        color_label = color_labels[g_color]
        type_label = type_labels[g_type]
        o_color_label = color_labels[g_o_color]
        o_type_label = type_labels[g_o_type]
        if g_o_pid == g_pid:
            acc_pid += 1
        if g_o_color == g_color:
            acc_color += 1
        if g_o_type == g_type:
            acc_type += 1
        # Create result filename
        #여기서 pid는 파일의 실제 값은 아니다.
        result_filename = f"{filename}_pid_{g_pid + 1}_{o_color_label}({color_label})_{o_type_label}({type_label}).jpg"
        # Save image to result directory
        shutil.copy2(g_impath, os.path.join(cfg.result, result_filename))
    print(f"Accuracy of pid: {acc_pid / total_images:.1%}")
    print(f"Accuracy of color: {acc_color / total_images:.1%}")
    print(f"Accuracy of type: {acc_type / total_images:.1%}")

def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


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
        '--transforms', type=str, nargs='+', help='data augmentation'
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
    reset_config(cfg, args)
    
    
    
    # Merge remaining options
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)
    
    # Store only the result directory path
    cfg.result = args.result
    
    # Store label file paths in a separate dictionary
    label_files = {
        'color': args.color_label,
        'type': args.type_label
    }

    log_name = 'inference.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)
    
    #model = load_model(cfg, datamanager)
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes= (datamanager.num_train_pids + datamanager.num_train_color_ids + datamanager.num_train_type_ids) if cfg.data.root == 'VeRi' else datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )

    #custom_load_pretrained_weights(model, cfg.model.load_weights)

    # Get classifier weights and biases from the model
        # 일반 nn.Module 모델인 경우
    if hasattr(model, 'classifier'):
        model_dict = model.state_dict()
        print(model_dict['classifier.weight'].shape)  # 또는
        print(model_dict['classifier.bias'].shape)
        classifier_weight = model.classifier.weight.data.clone()
        classifier_bias = model.classifier.bias.data.clone()

        model_dict = model.state_dict()
        model_dict['classifier.weight'] = classifier_weight
        model_dict['classifier.bias'] = classifier_bias
        model.load_state_dict(model_dict)
    num_params, flops = compute_model_complexity(
            model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))
    
    
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)
    
    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    model.eval()
    
    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )
    """
    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )
    """    
    # OSNet 모델을 래퍼로 감싸기
    print('Starting inference ...')

    

    inference(cfg, model, datamanager, label_files)


if __name__ == '__main__':
    main()
