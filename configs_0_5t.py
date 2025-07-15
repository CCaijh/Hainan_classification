train_configs={
     "num_classes":5,
     # "num_classes":3,
     "batch_size":80,
     "epochs":100,
      "lr":3e-4,
     "max_lr":0.003,
     "momentum":0.9,
     "weights":"",  # The pre-trained weight path should be set to an empty character if you do not want to load it
     "weight_decay":0.0001,
     "pct_start":0.1,
     "freeze_layers":False,
     "device":'cuda:0',
     "input_path": './HA419-IonogramSet-202505/HA_80_20_uniform',
     "output_path":'./logs',

     "seed": 2026,          # 固定随机种子
}
