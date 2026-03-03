import decimal
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer


class Trainer_Flow_Video(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Flow_Video, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Flow_Video")

        self.l1_loss = torch.nn.L1Loss()
        self.cycle_psnr_log = []
        self.mid_loss_log = []
        self.cycle_loss_log = []

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.cycle_psnr_log = mid_logs['cycle_psnr_log']
            self.mid_loss_log = mid_logs['mid_loss_log']
            self.cycle_loss_log = mid_logs['cycle_loss_log']

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().in_conv.parameters()},
                                {"params": self.model.get_model().extra_feat.parameters()},
                                # {"params": self.model.get_model().fusion_conv.parameters()},
                                {"params": self.model.get_model().fusion_conv1.parameters()},
                                {"params": self.model.get_model().fusion_conv2.parameters()},
                                {"params": self.model.get_model().fusion_conv3.parameters()},
                                {"params": self.model.get_model().fusion_conv4.parameters()},
                                {"params": self.model.get_model().fusion_conv5.parameters()},
                                {"params": self.model.get_model().msd_align.parameters()},
                                {"params": self.model.get_model().msdeformable_fusion.parameters()},
                                {"params": self.model.get_model().recons_net.parameters()},
                                {"params": self.model.get_model().upsample_layers.parameters()},
                                {"params": self.model.get_model().out_conv.parameters()},
                                # {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6},
                                {"params": self.model.get_model().kernel_net.parameters(), "lr": 1e-6},
                                {"params": self.model.get_model().cond_net.parameters()}],
                               **kwargs)
        return optimizer

    def train(self):
        print("Now training")
        # self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()   # 将模型设置为训练模式
        self.ckp.start_log()
        mid_loss_sum = 0.
        cycle_loss_sum = 0.

        for batch, (input, gt, _) in enumerate(self.loader_train):

            input = input.to(self.device)    # [8 5 3 64 64]
            gt_list = [gt[:, i, :, :, :].to(self.device) for i in range(self.args.n_sequence)]
            gt = gt_list[self.args.n_sequence // 2]  # 取中间帧

            output_dict, mid_loss = self.model({'x': input})
            output = output_dict['recons']
            kernel_list = output_dict['kernel_list']

            self.optimizer.zero_grad()  # 对优化器进行梯度初始化(将之前计算的梯度清零,避免梯度累积)

            loss = self.loss(output, gt)

            lr_cycle_list = [self.blur_down(g, k, self.args.scale) for g, k in zip(gt_list, kernel_list)]  # 模糊下采样操作
            cycle_loss = 0.
            for i, lr_cycle in enumerate(lr_cycle_list):
                cycle_loss = cycle_loss + self.l1_loss(lr_cycle, input[:, i, :, :, :])   # 循环一致性损失
            cycle_loss_sum = cycle_loss_sum + cycle_loss.item()
            loss = loss + cycle_loss

            if mid_loss:  # mid loss is the loss during the model  先判断是否存在中间损失
                loss = loss + self.args.mid_loss_weight * mid_loss
                mid_loss_sum = mid_loss_sum + mid_loss.item()

            loss.backward()  # 计算梯度(反向传播)
            self.optimizer.step()  # 更新模型参数

            self.ckp.report_log(loss.item())  # 记录当前批次的损失值到日志列表中

            if (batch + 1) % self.args.print_every == 0:         # batch从1开始每次循环+1,当99batch时，100 * 8 = 800
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[cycle: {:.4f}][mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    cycle_loss_sum / (batch + 1),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))   # 结束损失日志记录
        self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))  # 记录中间损失
        self.cycle_loss_log.append(cycle_loss_sum / len(self.loader_train)) # 记录循环一致性损失
        self.scheduler.step()  # 更新学习率调度器

    def test(self):    # 测试的时候batch_size = 1, 无patch_size
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()  # 将模型设置为评估模式
        self.ckp.start_log(train=False)
        cycle_psnr_list = []

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)  # tqdm函数，用于在命令行界面显示进度条，80是进度条的宽度
            for idx_img, (input, gt, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]   # 第一次循环：002.00000002

                input = input.to(self.device)  # [1 5 3 160 160]
                input_center = input[:, self.args.n_sequence // 2, :, :, :]  # 提取出中间帧作为输入中心帧 [1 3 160 160]
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device) # [1 3 640 640]

                output_dict, _ = self.model({'x': input})
                output = output_dict['recons']  # [1 3 640 640]
                kernel_list = output_dict['kernel_list']  # list * 5   [1 1 13 13]
                est_kernel = kernel_list[self.args.n_sequence // 2]

                lr_cycle_center = self.blur_down(gt, est_kernel, self.args.scale)

                cycle_PSNR = data_utils.calc_psnr(input_center, lr_cycle_center, rgb_range=self.args.rgb_range,
                                                  is_rgb=True)
                PSNR = data_utils.calc_psnr(gt, output, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)
                cycle_psnr_list.append(cycle_PSNR)

                if self.args.save_images:
                    gt, input_center, output, lr_cycle_center = data_utils.postprocess(
                        gt, input_center, output, lr_cycle_center,
                        rgb_range=self.args.rgb_range,
                        ycbcr_flag=False,
                        device=self.device)

                    est_kernel = self.process_kernel(est_kernel)

                    save_list = [gt, input_center, output, lr_cycle_center, est_kernel]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage Cycle-PSNR: {:.3f} PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                sum(cycle_psnr_list) / len(cycle_psnr_list),  # 计算周期 PSNR 值的平均值
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))         # 找到最佳的 PSNR 值和对应的训练轮数
            self.cycle_psnr_log.append(sum(cycle_psnr_list) / len(cycle_psnr_list))

            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
                self.ckp.plot_log(self.cycle_psnr_log, filename='cycle_psnr.pdf', title='Cycle PSNR')
                self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
                self.ckp.plot_log(self.cycle_loss_log, filename='cycle_loss.pdf', title='Cycle Loss')
                torch.save({
                    'cycle_psnr_log': self.cycle_psnr_log,
                    'mid_loss_log': self.mid_loss_log,
                    'cycle_loss_log': self.cycle_loss_log,
                }, os.path.join(self.ckp.dir, 'mid_logs.pt'))

    def conv_func(self, input, kernel, padding='same'):  # 实现了一个针对输入张量的每个通道进行卷积操作的函数，并返回卷积结果
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

    def blur_down(self, x, kernel, scale):  # 模糊和下采样的函数。首先，它将输入张量 x 进行模糊操作，然后对模糊结果进行下采样
        b, c, h, w = x.size()
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(self.conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        # down
        blurdown = blur[:, :, ::scale, ::scale]

        return blurdown

    def process_kernel(self, kernel):  # 对卷积核进行预处理
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)  # 将卷积核的数值范围归一化到 0-1 之间
        kernel = torch.cat([kernel, kernel, kernel], dim=1) # 将单通道的卷积核转换为 RGB 图像格式
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
