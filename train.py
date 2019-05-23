import os
from options.train_options import TrainOptions
from data.create_data_loader import create_data_loader
from models import create_model
from utils import visualizer, utils
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter


opt = TrainOptions().parse()

ckpt_dir = os.path.join(opt.ckpt_dir, opt.name)
writer = SummaryWriter(ckpt_dir)

# set data loader
print('### prepare DataLoader')
data_loader = create_data_loader(opt)
train_loader = data_loader.load_data()
print('training images = {}'.format(len(data_loader)))

# for iter, data in enumerate(train_loader):
#     # label = data['label'].repeat(1, 3, 1, 1)
#     # visualizer.show_loaded_image(data['color'], label, nrow=opt.batch_size)
#     visualizer.show_label(data['label'], nrow=opt.batch_size)

# set model
model = create_model(opt)

# training loop
print('### start training !')
for epoch in range(opt.epoch+opt.epoch_decay+1):
    for iter, data in enumerate(train_loader):
        model.set_variables(data)
        model.optimize_parameters()
        model.sum_epoch_losses()

        if iter % opt.print_iter_freq == 0:
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, iter, losses)

            saved_image = model.sample()
            visualizer.save_image(opt, epoch, saved_image)

    if epoch % opt.save_epoch_freq == 0:
        saved_image = model.sample()
        visualizer.save_image(opt, epoch, saved_image)

        model.save_networks(epoch)

    model.update_lr()

    for k, v in model.epoch_losses.items():
        writer.add_scalar(k, v/(iter+1), epoch)

    visualizer.print_epoch_losses(epoch, iter, model.epoch_losses)
    model.reset_epoch_losses()
