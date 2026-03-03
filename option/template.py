def set_template(args):
    if args.template == 'KernelPredict':
        args.task = "PretrainKernel"
        args.model = "Kernel"
        args.save = "Kernel_MANet_DBVSR_64_128_Pretrain"      # 保存文件的文件名
        args.data_train = 'REDS_ONLINE'  # 在线生成，即读一个GT生成一个LR，hrlr是本地生成好的
        # args.data_train = 'REDS_HRLR'  # 在线生成，即读一个GT生成一个LR，hrlr是本地生成好的
        args.dir_data = '../dataset/jilin189_BlurDown_Gaussian/'
        args.data_test = 'REDS_HRLR'  # 测试的时候就是本地生成的了，就是HRLR
        args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_002009010/'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5   # 每个训练样本包含 5 个序列帧，每个序列帧由连续的帧组成
        args.n_frames_per_video = 100  # 每个视频包含 100 帧
        args.est_ksize = 13
        args.loss = '1*L1'
        args.lr = 1e-4
        args.lr_decay = 20     # 指定每经过多少个轮次（epochs）进行一次学习速率衰减
        args.save_middle_models = True
        args.save_images = True
        args.epochs = 30
        args.batch_size = 16
        args.resume = True
        args.load = args.save

    elif args.template == 'VideoSR':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Ablation_1_large"
        # args.data_train = 'REDS_ONLINE'
        args.data_train = 'REDS_HRLR'
        args.dir_data = '../dataset/jilin189_BlurDown_Gaussian/'
        args.data_test = 'REDS_HRLR'
        # args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_002009010/'
        args.dir_data_test = '../dataset/val_20_BlurDown_Gaussian_001/'
        args.scale = 4
        args.patch_size = 64
        args.n_sequence = 5
        args.n_frames_per_video = 100
        args.n_feat = 128           # 特征的维度或通道数
        args.n_cond = 128           # 条件信息的维度或通道数
        args.est_ksize = 13
        args.extra_RBS = 3  # large:3   yuan = 3
        args.recons_RBS = 20  # large:20
        args.loss = '1*L1'
        # args.lr = 1e-4
        args.lr = 8e-5
        args.lr_decay = 50
        # args.lr_decay = 20 #############50
        args.save_middle_models = True
        args.save_images = False
        args.epochs = 100
        args.batch_size = 8
        # args.batch_size = 4
        # args.batch_size = 1

        # args.resume = True
        # args.load = args.save

    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))

