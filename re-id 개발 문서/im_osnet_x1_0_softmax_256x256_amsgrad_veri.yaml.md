model:
  name: 'osnet_x1_0'
  pretrained: True

data:
  type: 'image'
  sources: ['veri']
  targets: ['veri']
  height: 256
  width: 256
  combineall: False
  transforms: ['random_flip']
  save_dir: 'log/osnet_x1_0_veri_softmax'

loss:
  name: 'softmax'
  softmax:
    label_smooth: True

train:
  optim: 'amsgrad'
  lr: 0.0015
  max_epoch: 50
  batch_size: 64
  fixbase_epoch: 15
  open_layers: ['classifier']
  lr_scheduler: 'multi_step'
  stepsize: [25, 40]
  gamma: 0.1

test:
  batch_size: 64
  dist_metric: 'euclidean'
  normalize_feature: False
  evaluate: False
  eval_freq: 5
  rerank: False

## 주요 내용

training시 기본적으로 fixbase_epoch 까지는 open_layers에 설정된 layer만 training이 되고 나머지 layer는 train 되지 않는다.
그러므로 위 예를 보면 50 epoch 전체를 돌리는데, 15 epoch까지는 open_layers 에 설정한 **'classifier'** 만 training하고  그 이후로는 모든 layer를 training 한다는 의미이다. 
lr_scheduler이 'multi_step'  기본 single_step 이지만, multi_step으로 하면 learning rate가 training 시에 가변으로  바뀐다. stepsize에서 [25, 40] 은 25 epoch, 40 epoch 이후에 learning rate가 바뀐다는 의미이다. gamma 값은 얼마나 바뀔 것인가에 대한 비로 0.1이면 1/10씩 줄어 든다는 뜻이다.
eval_freq 값은 얼마나 자주 training시에 test를 돌려볼 것인지이다. 5이면 매 5에폭 마다 test를 돌려보겠다는 뜻이다.