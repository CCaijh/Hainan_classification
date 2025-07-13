train_configs={
     "num_classes":5,
     # "num_classes":3,
     "batch_size":80,
     "epochs":100,
      "lr":3e-4,
     "max_lr":0.003,
     "momentum":0.9,
     "weights":"",  # 预训练权重路径，如果不想载入就设置为空字符
     "weight_decay":0.0001,
     "pct_start":0.1,
     "freeze_layers":False,
     "device":'cuda:0',
     # "input_path":'Y:/ZMY/Hainan/Hainan_classification/modified_img/model_imgs',
     # 5type
     "input_path": 'Z:\ZQY\Hainan\Hainan_classification\HA_80_20_uniform',
     "output_path":'./logs',

     "seed": 56,          # 固定随机种子
}