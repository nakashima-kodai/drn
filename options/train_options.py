from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--save_epoch_freq', type=int, default=10)
        self.parser.add_argument('--print_iter_freq', type=int, default=100)
        self.parser.add_argument('--sample_dir', type=str, default='./sample')

        # training loop
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--epoch', type=int, default=100)
        self.parser.add_argument('--epoch_decay', type=int, default=100)

        # whole network
        self.parser.add_argument('--init_type', type=str, default='normal', choices=['normal', 'xaview', 'kaiming'])

        # optimizer
        self.parser.add_argument('--lr', type=float, default=0.01)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--weight_decay', type=float, default=0.0001)
        self.parser.add_argument('--lr_policy', type=str, default='step')
