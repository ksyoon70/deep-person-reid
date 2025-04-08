########################################
# vehicle_datamanager.py
########################################

from __future__ import division, print_function, absolute_import
import torch

from torchreid.data.sampler import build_train_sampler
from torchreid.data.datasets import init_image_dataset
from torchreid.data.datasets.dataset import Dataset
from torchreid.data.transforms import build_transforms
from torchreid.data.datamanager import DataManager


class VehicleImageDataManager(DataManager):
    r"""
    VehicleImageDataManager

    - ImageDataManager를 상속받아, 차량 데이터셋에 특화된 color_id, type_id 관련 정보를
      추가로 저장/처리하기 위한 매니저.
    - `trainset.num_train_color_ids`, `trainset.num_train_type_ids` 같은 속성이
      실제로 존재해야 합니다.
    """

    def __init__(
        self,
        root='',
        sources=None,
        targets=None,
        height=256,
        width=128,
        transforms='random_flip',
        k_tfm=1,
        norm_mean=None,
        norm_std=None,
        use_gpu=True,
        split_id=0,
        combineall=False,
        load_train_targets=False,
        batch_size_train=32,
        batch_size_test=32,
        workers=4,
        num_instances=4,
        num_cams=1,
        num_datasets=1,
        train_sampler='RandomSampler',
        train_sampler_t='RandomSampler',
        cuhk03_labeled=False,
        cuhk03_classic_split=False,
        market1501_500k=False
    ):
        # 먼저 부모 클래스(ImageDataManager)의 __init__을 호출
        super(VehicleImageDataManager, self).__init__(
            sources=sources,
            targets=targets,
            height=height,
            width=width,
            transforms=transforms,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_gpu=use_gpu
        )

        print('=> Loading train (source) dataset')
        trainset = []
        for name in self.sources:
            trainset_ = init_image_dataset(
                name,
                transform=self.transform_tr,
                k_tfm=k_tfm,
                mode='train',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            trainset.append(trainset_)
        trainset = sum(trainset)

        self._num_train_pids = trainset.num_train_pids
        self._num_train_cams = trainset.num_train_cams
      
        if hasattr(trainset, 'num_train_color_ids'):
            self._num_train_color_ids = trainset.num_train_color_ids
        else:
            # 존재하지 않을 경우 대비, 0 또는 None 할당
            self._num_train_color_ids = 0

        if hasattr(trainset, 'num_train_type_ids'):
            self._num_train_type_ids = trainset.num_train_type_ids
        else:
            self._num_train_type_ids = 0

        """
        torch.utils.data.DataLoader는 PyTorch에서 데이터를 로드하고, 이를 모델에 전달하기 위한 효율적인 방법을 제공하는 클래스입니다. 이 함수는 데이터셋을 쉽게 반복(iterate)할 수 있도록 여러 유용한 기능을 제공합니다. 주로 Dataset 객체와 함께 사용되며, 배치(batch) 처리, 셔플(shuffle), 병렬 처리 등을 지원합니다.
        
        주요 인자
        dataset: Dataset 객체를 받으며, 모델에 전달할 데이터셋을 지정합니다.
        batch_size: 한 번에 모델로 전달할 샘플의 수를 지정합니다.
        shuffle: True로 설정하면, 에포크마다 데이터를 무작위로 섞습니다.
        num_workers: 데이터를 로드할 때 사용할 병렬 작업자(worker) 수를 지정합니다.
        sampler: 데이터의 서브셋을 선택하거나 특정 샘플링 전략을 적용하는데 사용됩니다.
        pin_memory: True로 설정하면, 데이터 로드 속도를 높이기 위해 페이지 잠금 메모리를 사용합니다.
        drop_last: 배치 크기로 나누었을 때 남는 샘플이 있는 경우, 이를 버릴지(drop) 여부를 결정합니다.
        """
        self.train_loader = torch.utils.data.DataLoader(
            trainset,
            sampler=build_train_sampler(
                trainset.train,
                train_sampler,
                batch_size=batch_size_train,
                num_instances=num_instances,
                num_cams=num_cams,
                num_datasets=num_datasets
            ),
            batch_size=batch_size_train,
            shuffle=False,
            num_workers=workers,
            pin_memory=self.use_gpu,
            drop_last=True
        )

        self.train_loader_t = None
        if load_train_targets:
            # check if sources and targets are identical
            assert len(set(self.sources) & set(self.targets)) == 0, \
                'sources={} and targets={} must not have overlap'.format(self.sources, self.targets)

            print('=> Loading train (target) dataset')
            trainset_t = []
            for name in self.targets:
                trainset_t_ = init_image_dataset(
                    name,
                    transform=self.transform_tr,
                    k_tfm=k_tfm,
                    mode='train',
                    combineall=False, # only use the training data
                    root=root,
                    split_id=split_id,
                    cuhk03_labeled=cuhk03_labeled,
                    cuhk03_classic_split=cuhk03_classic_split,
                    market1501_500k=market1501_500k
                )
                trainset_t.append(trainset_t_)
            trainset_t = sum(trainset_t)

            self.train_loader_t = torch.utils.data.DataLoader(
                trainset_t,
                sampler=build_train_sampler(
                    trainset_t.train,
                    train_sampler_t,
                    batch_size=batch_size_train,
                    num_instances=num_instances,
                    num_cams=num_cams,
                    num_datasets=num_datasets
                ),
                batch_size=batch_size_train,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=True
            )

        print('=> Loading test (target) dataset')
        self.test_loader = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }
        self.test_dataset = {
            name: {
                'query': None,
                'gallery': None
            }
            for name in self.targets
        }

        for name in self.targets:
            # build query loader
            queryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='query',
                combineall=combineall,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['query'] = torch.utils.data.DataLoader(
                queryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            # build gallery loader
            galleryset = init_image_dataset(
                name,
                transform=self.transform_te,
                mode='gallery',
                combineall=combineall,
                verbose=False,
                root=root,
                split_id=split_id,
                cuhk03_labeled=cuhk03_labeled,
                cuhk03_classic_split=cuhk03_classic_split,
                market1501_500k=market1501_500k
            )
            self.test_loader[name]['gallery'] = torch.utils.data.DataLoader(
                galleryset,
                batch_size=batch_size_test,
                shuffle=False,
                num_workers=workers,
                pin_memory=self.use_gpu,
                drop_last=False
            )

            self.test_dataset[name]['query'] = queryset.query
            self.test_dataset[name]['gallery'] = galleryset.gallery

        print('\n')
        print('  **************** Summary ****************')
        print('  source            : {}'.format(self.sources))
        print('  # source datasets : {}'.format(len(self.sources)))
        print('  # source ids      : {}'.format(self.num_train_pids))
        print('  # source images   : {}'.format(len(trainset)))
        print('  # source cameras  : {}'.format(self.num_train_cams))
        if load_train_targets:
            print(
                '  # target images   : {} (unlabeled)'.format(len(trainset_t))
            )
        print('   # source color IDs (train) : {}'.format(self._num_train_color_ids))
        print('   # source type IDs (train)  : {}'.format(self._num_train_type_ids))
        print('  target            : {}'.format(self.targets))
        print('  *****************************************')
        print('\n')
        # 확인차 출력 (원한다면 요약 정보에 추가)
        print('=> VehicleImageDataManager init complete')
        
         

    @property
    def num_train_color_ids(self):
        """Returns the number of training color IDs."""
        return self._num_train_color_ids

    @property
    def num_train_type_ids(self):
        """Returns the number of training type IDs."""
        return self._num_train_type_ids
    
    
