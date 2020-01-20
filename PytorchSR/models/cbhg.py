import torch.nn as nn
import numpy as np

from data.data_utils import PHNS
from models.model import Model, Model1
from models.modules import Prenet, CBHG, SeqLinear
from settings.hparam import hparam as hp


class CBHGNet(Model):

    def __init__(self):
        super().__init__()
        self.prenet = Prenet(hp.default.n_mfcc, hp.train.hidden_units,
                             hp.train.hidden_units // 2, dropout_rate=hp.train.dropout_rate)
        self.cbhg = CBHG(
            K=hp.train.num_banks,
            hidden_size=hp.train.hidden_units // 2,
            num_highway_blocks=hp.train.num_highway_blocks,
            num_gru_layers=1
        )
        self.output = SeqLinear(hp.train.hidden_units, len(PHNS))

    def forward(self, x):
        print("Old situation here")
        x = x.contiguous().transpose(1, 2)
        print(np.shape(x))
        net = self.prenet(x)
        net, _ = self.cbhg(net)
        logits_ppg = self.output(net)
        return logits_ppg

class CBHGNet1(Model1):

    def __init__(self):
        super().__init__()
        self.prenet = Prenet(hp.default.n_mfcc, hp.train1.hidden_units,
                             hp.train1.hidden_units // 2, dropout_rate=hp.train1.dropout_rate)
        self.cbhg = CBHG(
            K=hp.train.num_banks,
            hidden_size=hp.train.hidden_units // 2,
            num_highway_blocks=hp.train.num_highway_blocks,
            num_gru_layers=1
        )
        self.output = SeqLinear(hp.train1.hidden_units, len(PHNS))

    def forward(self, x, y_mel, y_spec):
        print("I am entering here - syntehsiser")
        print(np.shape(x))
        print(np.shape(y_mel))
        print(np.shape(y_spec))
        x = x.contiguous().transpose(1, 2)
        print(np.shape(x))
        net = self.prenet(x)
        print(net)
        pred_mel, _ = self.cbhg(net)
        print(np.shape(pred_mel))
        pred_mel = self.output(pred_mel,y_mel.shape[-1])
        pred_spec = self.output(pred_mel, hp.train.hidden_units // 2)
        pred_spec = self.cbhg(pred_spec)
        pred_spec = self.output(pred_spec,y_spec.shape[-1])

        return pred_mel, pred_spec