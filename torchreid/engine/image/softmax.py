from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import CrossEntropyLoss

from ..engine import Engine

import numpy as np
import matplotlib.pyplot as plt

class ImageSoftmaxEngine(Engine):
    r"""Softmax-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='softmax'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageSoftmaxEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-softmax-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageSoftmaxEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids + self.datamanager.num_color_ids + self.datamanager.num_type_ids if self.datamanager.targets[0] == 'veri' and self.datamanager.output_usage == 'mixture' else self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        if self.datamanager.targets[0] == 'veri':
            imgs, pids,colors, typeids = self.parse_data_for_train(data)
        else:
            imgs, pids = self.parse_data_for_train(data)
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
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()
            if self.datamanager.targets[0] == 'veri':
                colors = colors.cuda()
                typeids = typeids.cuda()
        outputs = self.model(imgs)
        if hasattr(self.model.module, 'classifier'):
            model_dict = self.model.state_dict()
            #print('model_dict[module.classifier.weight][0,0:5] :{}'.format(model_dict['module.classifier.weight'][0,0:5]))
        
        if self.datamanager.targets[0] == 'veri':
            # 모델 결과(outputs)가 (batch_size, 594)라고 가정
            # 슬라이싱 인덱스 주의: 0~574까지 pid, 575~584까지 color, 585~593까지 type
            # pid: 0 ~ (575-1), color: 575 ~ (575+10-1)=584, type: 585 ~ (585+9-1)=593
            pid_end = self.datamanager.num_train_pids                 # 575
            color_end = pid_end + self.datamanager.num_color_ids # 575 + 10 = 585
            logits_pid   = outputs[:, :pid_end]        # shape (B, 575)
            logits_color = outputs[:, pid_end :color_end]     # shape (B, 10)
            logits_type  = outputs[:, color_end:]        # shape (B, 9)
            # dictionary 형태로 담아주면 이후 사용하기 편함
            output = {
                'pid': logits_pid,
                'color': logits_color,
                'type': logits_type
            }

            
            if self.datamanager.output_usage == 'mixture':
                loss_pid = self.compute_loss(self.criterion, output['pid'], pids)
                loss_color = self.compute_loss(self.criterion, output['color'], colors)
                loss_type = self.compute_loss(self.criterion, output['type'], typeids)
                loss = loss_pid + loss_color + loss_type
            else:
                loss_pid = self.compute_loss(self.criterion, output['pid'], pids)
                loss = loss_pid # pid만 사용하는 경우
        else:
            loss = self.compute_loss(self.criterion, outputs, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6) 로그/모니터링용 loss_summary
        if self.datamanager.targets[0] == 'veri':
            
            if self.datamanager.output_usage == 'mixture':
                acc_pid = metrics.accuracy(output['pid'], pids)[0].item()
                acc_color = metrics.accuracy(output['color'], colors)[0].item()
                acc_type = metrics.accuracy(output['type'], typeids)[0].item()
                acc_total = acc_pid + acc_color + acc_type
                acc_mean = acc_total / 3.0  # 세 accuracy의 평균
            else:
                acc_pid = metrics.accuracy(output['pid'], pids)[0].item()
                acc_mean = acc_pid # pid만 사용하는 경우
            loss_summary = {
                'loss': loss.item(),
                'acc': acc_mean
            }
        else:
            acc = metrics.accuracy(outputs, pids)[0].item()
            loss_summary = {
                'loss': loss.item(),
                'acc': acc
            }

        return loss_summary
