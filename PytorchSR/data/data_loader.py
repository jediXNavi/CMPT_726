import torch
from data.data_utils import get_wav_data, get_mfccs_phones, get_mfccs_and_spectrogram, load_data
from settings.hparam import hparam as hp
from torch.utils.data import Dataset

from torch import FloatTensor, LongTensor


class VoiceData:

    def __init__(self, wav_file_path, mode='train', init_all=True):
        if mode not in ['train', 'test']:
            raise NotImplementedError('Mode %s is not implemented ! You can use train or test' % mode)
        self.wav_file_path = wav_file_path
        self.wav_data = get_wav_data(self.wav_file_path)
        self.mode = mode
        self.mfccs = None
        self.phns = None
        self.mels = None
        self.spec = None
        if init_all:
            self.init_data()

    @property
    def phn_file_path(self):
        if self.wav_file_path.endswith('arr'):
            return self.wav_file_path.replace('voice_all_arr/%s' % self.mode, 'voice_all').replace("WAV.wav.arr",
                                                                                                   "PHN").replace(
                "wav.arr", "PHN")
        else:
            return self.wav_file_path.replace("WAV", "PHN").replace("wav", "phn")

    @property
    def phn_length(self):
        return int(hp.default.duration // hp.default.frame_shift + 1)

    def init_data(self):
        is_random_crop = 'train' == self.mode
        # train 1 or test 1
        self.mfccs, self.phns = get_mfccs_phones(self.wav_data, self.phn_file_path,
                                                 random_crop=is_random_crop, trim=False)
        return self

    def mfcc(self):
        if self.mfccs is None:
            raise RuntimeError('Mfcc is not initialized !!!')
        return self.mfccs

    def phn(self):
        if self.phns is None:
            raise RuntimeError('Phoneme is not initialized !!!')
        return self.phns

    def wav(self):
        if self.wav_data is None:
            raise RuntimeError('Voice Data is not initialized !!!')
        return self.wav_data

    def mel(self):
        if self.mels is None:
            raise RuntimeError('Mel is not initialized !!!')
        return self.mels

    def spectrogram(self):
        if self.spec is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.spec

class VoiceData1:

    def __init__(self, wav_file_path, mode='train1', init_all=True):
        if mode not in ['train','train1','test','test1']:
            raise NotImplementedError('Mode %s is not implemented ! You can use train or test' % mode)
        self.wav_file_path = wav_file_path
        self.wav_data = get_wav_data(self.wav_file_path)
        self.mode = mode
        self.mfccs = None
        self.phns = None
        self.mels = None
        self.spec = None
        self.mag_db = None
        self.mel_db = None
        if init_all:
            self.init_data()

    @property
    def phn_length(self):
        return int(hp.default.duration // hp.default.frame_shift + 1)

    def init_data(self):
        is_random_crop = 'train1' == self.mode
        # train 1 or test 1
        self.mfccs, self.mag_db, self.mel_db = get_mfccs_and_spectrogram(self.wav_data, hp.default.win_length,
                                                                         hp.default.hop_length,
                                                 random_crop=is_random_crop, trim=False,duration=hp.default.duration)
        return self

    def mfcc(self):
        if self.mfccs is None:
            raise RuntimeError('Mfcc is not initialized !!!')
        return self.mfccs

    def phn(self):
        if self.phns is None:
            raise RuntimeError('Phoneme is not initialized !!!')
        return self.phns

    def wav(self):
        if self.wav_data is None:
            raise RuntimeError('Voice Data is not initialized !!!')
        return self.wav_data

    def mel(self):
        if self.mels is None:
            raise RuntimeError('Mel is not initialized !!!')
        return self.mels

    def spectrogram(self):
        if self.spec is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.spec

    def mfccs(self):
        if self.mfccs is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.mfccs

    def mag_db(self):
        if self.mag_db is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.mag_db

    def mel_db(self):
        if self.mel_db is None:
            raise RuntimeError('Spectrogram is not initialized !!!')
        return self.mel_db

class VoiceDataset(Dataset):

    def __init__(self, mode='train', data_split=-1.0, init_all=True):
        self.mode = mode
        self.wav_files = load_data(mode=mode, split=data_split)
        self.idx_list = list(range(len(self.wav_files)))
        self.init_all = init_all
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, idx):
        wav_file_path = self.wav_files[idx]
        voice_data = VoiceData(wav_file_path, self.mode, init_all=self.init_all)
        return FloatTensor(voice_data.mfcc()), LongTensor(voice_data.phn())

    def __len__(self):
        return len(self.idx_list)

class VoiceDataset1(Dataset):

    def __init__(self, mode='train1', data_split=-1.0, init_all=True):
        self.mode = mode
        self.wav_files = load_data(mode=mode, split=data_split)
        self.idx_list = list(range(len(self.wav_files)))
        self.init_all = init_all
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, idx):
        wav_file_path = self.wav_files[idx]
        voice_data1 = VoiceData1(wav_file_path, self.mode, init_all=self.init_all)
        return FloatTensor(voice_data1.mfccs), FloatTensor(voice_data1.mag_db), FloatTensor(voice_data1.mel_db)

    def __len__(self):
        return len(self.idx_list)

class TrainVoiceDataset(VoiceDataset):

    def __init__(self, init_all=True):
        super().__init__(
            mode='train',
            init_all=init_all
        )


class TestVoiceDataset(VoiceDataset):

    def __init__(self, init_all=True):
        super().__init__(
            mode='test',
            init_all=init_all
        )

class TrainVoiceDataset1(VoiceDataset1):

    def __init__(self, init_all=True):
        super().__init__(
            mode='train1',
            init_all=init_all
        )


class TestVoiceDataset1(VoiceDataset1):

    def __init__(self, init_all=True):
        super().__init__(
            mode='test1',
            init_all=init_all
        )