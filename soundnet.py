import numpy as np
import torch
import torch.nn as nn
from torch import bmm, cat, randn, zeros
from torch.autograd import Variable
import os

from goggles.torch_soundnet.util import load_from_txt

LEN_WAVEFORM = 22050 * 20

local_config = {
    'batch_size': 1,
    'eps': 1e-5,
    'sample_rate': 22050,
    'load_size': 22050 * 20,
    'name_scope': 'SoundNet_TF',
    'phase': 'extract',
}

class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(64, 1), stride=(2, 1), padding=(32, 0)),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8, 1), stride=(8, 1)),

            nn.Conv2d(16, 32, kernel_size=(32, 1), stride=(2, 1), padding=(16, 0)),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((8, 1), stride=(8, 1)),

            nn.Conv2d(32, 64, kernel_size=(16, 1), stride=(2, 1), padding=(8, 0)),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(8, 1), stride=(2, 1), padding=(4, 0)),
            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1), stride=(4, 1)),

            nn.Conv2d(256, 512, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 1024, kernel_size=(4, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(1024, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        # self.embedding = nn.Sequential(
        #     nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1)),
        #     nn.Conv2d(1024, 401, kernel_size=(8, 1), stride=(2, 1)),
        # )
        self.conv8_objs = nn.Conv2d(1024, 1000, kernel_size=(8, 1), stride=(2, 1))
        self.conv8_scns = nn.Conv2d(1024, 401, kernel_size=(8, 1), stride=(2, 1))

    def forward(self, waveform):
        """
            Args:
                waveform (Variable): Raw 20s waveform.
        """
        if torch.cuda.is_available():
            waveform.cuda()

        out = self.conv1(waveform)
        print('Max value of conv1: {:.4f}'.format(np.max(out.data.numpy())))
        print('Min value of conv1: {:.4f}'.format(np.min(out.data.numpy())))
        out = self.batchnorm1(out)
        print('Max value of BN1: {:.4f}'.format(np.max(out.data.numpy())))
        print('Min value of BN1: {:.4f}'.format(np.min(out.data.numpy())))
        out = self.relu1(out)
        print('Max value of relU1: {:.4f}'.format(np.max(out.data.numpy())))
        print('Min value of relu1: {:.4f}'.format(np.min(out.data.numpy())))
        out = self.maxpool1(out)
        print('Max value of maxpool1: {:.4f}'.format(np.max(out.data.numpy())))
        print('Min value of maxpool1: {:.4f}'.format(np.min(out.data.numpy())))

        return out.data.numpy()

    @staticmethod
    def put_weights(batchnorm, conv, params_w, batch_norm=True):
        if batch_norm:
            bn_bs = params_w['beta']
            batchnorm.bias.data = torch.from_numpy(bn_bs)
            bn_ws = params_w['gamma']
            batchnorm.weight.data = torch.from_numpy(bn_ws)
            bn_mean = params_w['mean']
            batchnorm.running_mean.data = torch.from_numpy(bn_mean)
            bn_var = params_w['var']
            batchnorm.running_var.data = torch.from_numpy(bn_var)

        conv_bs = params_w['biases']
        conv.bias.data = torch.from_numpy(conv_bs)
        conv_ws = params_w['weights']
        conv.weight.data = torch.from_numpy(conv_ws).permute(3, 2, 0, 1)
        return batchnorm, conv

    def load_weights(self, batch_norm=False):
        file_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.abspath(os.path.join(file_path, "models/sound8.npy"))
        param_G = np.load(path, encoding='latin1', allow_pickle=True).item()
        params_w = param_G['conv1']
        self.features[1], self.features[0] = self.put_weights(self.features[1], self.features[0], params_w, batch_norm=batch_norm)

        params_w = param_G['conv2']
        self.features[5], self.features[4] = self.put_weights(self.features[5], self.features[4], params_w, batch_norm=batch_norm)

        params_w = param_G['conv3']
        self.features[9], self.features[8] = self.put_weights(self.features[9], self.features[8], params_w, batch_norm=batch_norm)

        params_w = param_G['conv4']
        self.features[12], self.features[11] = self.put_weights(self.features[12], self.features[11], params_w, batch_norm=batch_norm)

        params_w = param_G['conv5']
        self.features[15], self.features[14] = self.put_weights(self.features[15], self.features[14], params_w, batch_norm=batch_norm)

        params_w = param_G['conv6']
        self.features[19], self.features[18] = self.put_weights(self.features[19], self.features[18], params_w, batch_norm=batch_norm)

        params_w = param_G['conv7']
        self.features[22], self.features[21] = self.put_weights(self.features[22], self.features[21], params_w, batch_norm=batch_norm)

        params_w = param_G['conv8']
        _, self.conv8_objs = self.put_weights([], self.conv8_objs, params_w, batch_norm=False)

        params_w = param_G['conv8_2']
        _, self.conv8_scns = self.put_weights([], self.conv8_scns, params_w, batch_norm=False)


def get_model(pretrained=True):
    pytorch_model = SoundNet()
    if pretrained:
        pytorch_model.load_weights(batch_norm=True)
    return pytorch_model

# @mem.cache()
def extract_features():
    audio_txt = 'audio_files.txt'
    model = SoundNet()
    model.load_weights()
    # Extract Feature
    sound_samples, audio_paths = load_from_txt(audio_txt, config=local_config)
    print(LEN_WAVEFORM / 6)
    print(model)
    features = {}
    features['feats'] = []
    features['paths'] = []
    model.eval()
    for idx, sound_sample in enumerate(sound_samples):
        print(audio_paths[idx])
        new_sample = torch.from_numpy(sound_sample)
        output = model.forward(new_sample)
        features['feats'].append(output)
        features['paths'].append(audio_paths[idx])
    return features

if __name__ == '__main__':
    extract_features()
