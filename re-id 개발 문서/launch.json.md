{

    "version": "0.2.0",

    "configurations": [    

        {

            "name": "Python: Current File",

            "type": "python",

            "request": "launch",

            "program": "${file}",

            "console": "integratedTerminal",

            "args": [

                "--train","true",

                "--config-file", "configs/im_osnet_x1_0_softmax_256x256_amsgrad_veri.yaml",

                "--transforms", "random_flip","random_erase",                

                "-s", "veri",              

                "-t", "veri",

                "--root", "VeRi",

                "--result", "VeRi/veri/results",

                "--color_label", "VeRi/veri/list_color.txt",

                "--type_label", "VeRi/veri/list_type.txt",

                "--output_usage", "feature",

                "model.load_weights", "log/osnet_x1_0_veri_softmax/model/model.pth.tar-70",

                "model.resume", "log/osnet_x1_0_veri_softmax/model/model.pth.tar-70"

            ]

        }

    ]

}

## 항목설명
- train  : train 시에는 true inference 할 때는 false가 된다.
- config_file : training 할 때 참조 할 osnet의 configuration file 이다. osnet이 아니고 resnet이면 다른 파일로 바꾸면 된다.
- color_label : color 라벨링 값이 있는 파일의 경로이다.
- type_label : 차량 type 라벨링 값이 있는 파일의 경로이다.
- output_usage : **"feature"** 이면 feature vector만을 training 하고 **"mixture"** 라고 하면 feature에 추가로 color와 type까지 training을 한다.
- model.load_weights : inference 시에 읽을 weight file 경로이다.
- model.resume : 계속하여 training 할때 읽을 weight file 경로이다.
- s, t : veri dataset 을 training 할 때 사용하는 값이다. "veri" 로 두면 된다.