import fire
import utils
import torch
from torch import optim

from models.model import Model
from settings.hparam import hparam as hp


class Runner:
    IMPLEMENTED_MODELS = ['cbhg', 'mgru']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PATH = '../savemodel.pth'

    def train(self, model, checkpoint='', is_cuda=True, is_multi_gpu=True, logdir='', savedir=''):
        if model not in self.IMPLEMENTED_MODELS:
            raise NotImplementedError('%s model is not implemented !' % model)

        mode = 'train'
        logger = utils.get_logger(mode)
        device = self.device

        # initialize hyperparameters
        hp.set_hparam_yaml(mode)
        logger.info('Setup mode as %s, model : %s' % (mode, model))

        # get network
        network = utils.get_networks(device, model, checkpoint, is_cuda, is_multi_gpu)

        # setup dataset
        train_dataloader = Model.data_loader(mode='train')
        test_dataloader = Model.data_loader(mode='test')

        # setup optimizer:
        parameters = network.parameters()
        logger.info(network)
        lr = getattr(hp, mode).lr
        optimizer = optim.Adam([p for p in parameters if p.requires_grad], lr=lr)

        # pass model, loss, optimizer and dataset to the trainer
        # get trainer
        trainer = utils.get_trainer()(network, optimizer, train_dataloader, test_dataloader, is_cuda, logdir, savedir)

        # train!
        trainer.run(hp.train.num_epochs)

        torch.save(network.state_dict(),self.PATH)

    def eval(self):
        raise NotImplementedError('Evaluation mode is not implemented!')


if __name__ == '__main__':
    fire.Fire(Runner)