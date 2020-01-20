import fire
import utils
import torch
from torch import optim

from models.model import Model, Model1
from settings.hparam import hparam as hp


class Runner1:
    IMPLEMENTED_MODELS = ['cbhg', 'synthesizer']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = '../savemodel.pth'
    CHECKPOINT_PATH = '../checkpoint'

    def train1(self, model, checkpoint='', is_cuda=True, is_multi_gpu=True, logdir='', savedir=''):
        if model not in self.IMPLEMENTED_MODELS:
            raise NotImplementedError('%s model is not implemented !' % model)

        mode = 'train1'
        logger = utils.get_logger(mode)
        device = self.device

        # initialize hyperparameters
        hp.set_hparam_yaml(mode)
        logger.info('Setup mode as %s, model : %s' % (mode, model))

        # get network
        network = utils.get_networks(device, model, checkpoint, is_cuda, is_multi_gpu)
        network1 = utils.get_networks1(device, model, checkpoint, is_cuda, is_multi_gpu)

        # setup dataset
        train_dataloader = Model1.data_loader(mode='train1')
        test_dataloader = Model1.data_loader(mode='test1')

        # setup optimizer:
        parameters = network1.parameters()
        logger.info(network)
        lr = getattr(hp, mode).lr
        optimizer = optim.Adam([p for p in parameters if p.requires_grad], lr=lr,amsgrad=True)

        # pass model, loss, optimizer and dataset to the trainer
        # get trainer
        trainer = utils.get_trainer1('synthesizer')(network, network1, optimizer, train_dataloader, test_dataloader, is_cuda, logdir, savedir)

        # train!
        trainer.run(hp.train.num_epochs)

        torch.save({
            'net': network1.state_dict(),
            'optimizer': optimizer.state_dict()}, self.CHECKPOINT_PATH)

        torch.save(network1,self.MODEL_PATH)

    def eval(self):
        raise NotImplementedError('Evaluation mode is not implemented!')


if __name__ == '__main__':
    fire.Fire(Runner1)
