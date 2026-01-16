import sys
import time
import os.path as osp
import argparse
import torch
import torch.nn as nn

import torchreid
from torchreid.data.vehicle_datamanager import VehicleImageDataManager
from torchreid.data.ocr_datamanager import OcrImageDataManager
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)

from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)


def build_datamanager(cfg):
    if cfg.data.type == 'image':
        if cfg.data.root == 'VeRi':  # VeRi 데이터셋인 경우
            return torchreid.data.VehicleImageDataManager(**imagedata_kwargs(cfg))
        else:
            return torchreid.data.ImageDataManager(**imagedata_kwargs(cfg))
    else:
        return torchreid.data.VideoDataManager(**videodata_kwargs(cfg))


def load_model(cfg, datamanager):
    print('Building model: {}'.format(cfg.model.name))
    model = torchreid.models.build_model(
        name=cfg.model.name,
        num_classes=datamanager.num_train_pids,
        loss=cfg.loss.name,
        pretrained=cfg.model.pretrained,
        use_gpu=cfg.use_gpu
    )
    
    if cfg.model.load_weights and check_isfile(cfg.model.load_weights):
        load_pretrained_weights(model, cfg.model.load_weights)

    if cfg.use_gpu:
        model = nn.DataParallel(model).cuda()
    
    model.eval()  # Set the model to evaluation mode for inference
    return model


def inference(cfg, model, datamanager):
    # You can implement the inference logic here based on your requirements
    # For example, you can loop through the test data and perform forward passes
    targets = list(datamanager.test_loader.keys())

    for name in targets:
        domain = 'source' if name in datamanager.sources else 'target'
        print('##### Evaluating {} ({}) #####'.format(name, domain))
        query_loader = datamanager.test_loader[name]['query']
        gallery_loader = datamanager.test_loader[name]['gallery']
        break

    # Extract features for query images
    query_features, query_pids, query_camids = torchreid.utils.extract_features(
        model, query_loader, use_gpu=cfg.use_gpu
    )
    
    # Extract features for gallery images
    gallery_features, gallery_pids, gallery_camids = torchreid.utils.extract_features(
        model, gallery_loader, use_gpu=cfg.use_gpu
    )

    # Compute distance matrix and evaluate
    distmat = torchreid.utils.compute_distance_matrix(query_features, gallery_features, metric=cfg.test.distance_metric)
    torchreid.utils.evaluate_rank(
        distmat, query_pids, gallery_pids, query_camids, gallery_camids
    )


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
    cfg.merge_from_list(args.opts)
    set_random_seed(cfg.train.seed)
    check_cfg(cfg)

    log_name = 'inference.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.data.save_dir, log_name))

    print('Show configuration\n{}\n'.format(cfg))
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    if cfg.use_gpu:
        torch.backends.cudnn.benchmark = True

    datamanager = build_datamanager(cfg)

    model = load_model(cfg, datamanager)

    print('Starting inference ...')
    inference(cfg, model, datamanager)


if __name__ == '__main__':
    main()
