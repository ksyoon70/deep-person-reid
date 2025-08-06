import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights,load_checkpoint ,compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs, veri_imagedata_kwargs
)
from collections import OrderedDict

def build_datamanager(cfg):
    if cfg.data.type == 'image':
        if cfg.data.root == 'VeRi':
            return torchreid.data.VehicleImageDataManager(**veri_imagedata_kwargs(cfg))
        else:
            return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def build_engine(cfg, datamanager, model, optimizer, scheduler):
    if cfg.data.type == 'image':
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.ImageSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

        else:
            engine = torchreid.engine.ImageTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    else:
        if cfg.loss.name == 'softmax':
            engine = torchreid.engine.VideoSoftmaxEngine(
                datamanager,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth,
                pooling_method=cfg.video.pooling_method
            )

        else:
            engine = torchreid.engine.VideoTripletEngine(
                datamanager,
                model,
                optimizer=optimizer,
                margin=cfg.loss.triplet.margin,
                weight_t=cfg.loss.triplet.weight_t,
                weight_x=cfg.loss.triplet.weight_x,
                scheduler=scheduler,
                use_gpu=cfg.use_gpu,
                label_smooth=cfg.loss.softmax.label_smooth
            )

    return engine


def reset_config(cfg, args):
    if args.root:
        cfg.data.root = args.root
    if args.sources:
        cfg.data.sources = args.sources
    if args.targets:
        cfg.data.targets = args.targets
    if args.transforms:
        cfg.data.transforms = args.transforms
    if args.color_label:        # color label, type label 경로 추가 by 윤경섭 25.05.07
        cfg.veri.color_class_filename = args.color_label
    if args.type_label:
        cfg.veri.type_class_filename = args.type_label
    if args.output_usage:
        cfg.veri.output_usage = args.output_usage # feature or mixture by 윤경섭 25.08.01


def check_cfg(cfg):
    if cfg.loss.name == 'triplet' and cfg.loss.triplet.weight_x == 0:
        assert cfg.train.fixbase_epoch == 0, \
            'The output of classifier is not included in the computational graph'


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--train',      #train 여부의 옵션. True 로 하면 train하고, False 로 하면 inference함
                    type=lambda x: x.lower() in ['true', '1', 'yes'],
                    default=False)
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
    #---------------이 부분은 필요가 없지만 공통 영역으로 넣어 두었다. -----------
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
        '--output_usage', type=str, default='feature', help='model output usage (feature or mixture)'
    )
    #-----------------------------------------------------------------------
    #이후의 부분은 옵션 처리
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
        cfg.merge_from_file(args.config_file) # load values from a file
    reset_config(cfg, args)
    if args.opts:
        cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'test.log' if cfg.test.evaluate else 'train.log'  # configfile xxxx.yaml 내 test: evaluate: True/False 에 따라 train.log 또는 test.log로 설정
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        #veri 데이터셋인 경우이고 output_usage가 mixture 이면 색·종류 레퍼런스 추가 하고 그 외에는 pid만 사용
        num_classes= (datamanager.num_train_pids + datamanager.num_color_ids + datamanager.num_type_ids) if cfg.data.root == 'VeRi' and cfg.veri.output_usage == 'mixture' else datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
       **{'use_gpu':cfg.use_gpu, 'skip_classifier': False if cfg.data.root == 'VeRi' and cfg.veri.output_usage == 'mixture' else True } # cfg를 추가로 넘겨 준다. 25.05.07
    )

    num_params, flops = compute_model_complexity(
        model, (1, 3, cfg.data.height, cfg.data.width)
    )
    print('Model complexity: params={:,} flops={:,}'.format(num_params, flops))

    if args.train  == False: #이건, inference에 쓰는 코드
        if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
            load_pretrained_weights(model, cfg.model.load_weights)


    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()

    optimizer = torchreid.optim.build_optimizer(model, **optimizer_kwargs(cfg))
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, **lr_scheduler_kwargs(cfg)
    )
    # 당분간 쓰진 않느다.
    """
    if cfg.model.resume and check_isfile(cfg.model.resume):
        cfg.train.start_epoch = resume_from_checkpoint(
            cfg.model.resume, model, optimizer=optimizer, scheduler=scheduler
        )
    """
    print(
        'Building {}-engine for {}-reid'.format(cfg.loss.name, cfg.data.type)
    )
    engine = build_engine(cfg, datamanager, model, optimizer, scheduler)
    engine.run(**engine_run_kwargs(cfg))


if __name__ == '__main__':
    main()
