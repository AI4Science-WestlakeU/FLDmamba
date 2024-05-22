import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, S_Mamba, \
    Flashformer_M, Flowformer_M, Autoformer, Autoformer_M, Transformer_M, \
    Informer_M, Reformer_M
import sys
# sys.path.append('/root/mamba/S-D-Mamba-main/mamba_fft')
import model
print(model.__file__)
from experiments.mamba_fft import Mamba_fft,Mamba_simple
import pdb

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,

            'Transformer': Transformer,
            'Transformer_M': Transformer_M,

            'Informer': Informer,
            'Informer_M': Informer_M,

            'Reformer': Reformer,
            'Reformer_M': Reformer_M,

            'Flowformer': Flowformer,
            'Flashformer_M': Flashformer_M,

            'Flashformer': Flashformer,
            'Flowformer_M': Flowformer_M,

            'Autoformer': Autoformer,
            'Autoformer_M': Autoformer_M,

            'S_Mamba': S_Mamba,
            'Mamba_fft':Mamba_fft,
            'Mamba_simple':Mamba_simple
        }
 
        
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f'Total number of parameters: {total_params}')
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
