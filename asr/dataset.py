import csv
import os
import pickle
import xml.dom.minidom

import numpy
import torch
import torch.utils.data

from .feature import extract_feature_from_wavfile


class CSJTalk(object):

    def __init__(self, talk_id, talk_category, core_or_noncore):
        self.id = talk_id
        self.category = talk_category
        self.is_monoral = False if talk_category == '対話' else True
        self.is_core = True if core_or_noncore == 'コア' else False
        self.core_dirname = 'core' if self.is_core is True else 'noncore'
        self._feature = {'L': None, 'R': None}
        self._xml_content = None

    def load_xml(self, xml_dir):
        xml_filename = '{}.xml'.format(self.id)
        xml_filepath = os.path.join(xml_dir, self.core_dirname, xml_filename)
        self._xml_content = xml.dom.minidom.parse(xml_filepath)

    def get_xml(self):
        if self._xml_content is None:
            raise ValueError('xml is not loaded. Call load_xml method first.')
        return self._xml_content

    def extract_feature(self, wav_dir, feature_params):
        """
        Args:
            wav_dir (str): WAV directory
            feature_params (asr.feature.FeatureParams): FeatureParams object
        """
        if self.is_monoral:
            wav_filename = '{}.wav'.format(self.id)
            wav_filepath = os.path.join(
                wav_dir, self.core_dirname, wav_filename)
            self._feature['L'] = extract_feature_from_wavfile(
                wav_filepath, feature_params)
        else:
            for channel in ('L', 'R'):
                wav_filename = '{}-{}.wav'.format(self.id, channel)
                wav_filepath = os.path.join(
                    wav_dir, self.core_dirname, wav_filename)
                self._feature[channel] = extract_feature_from_wavfile(
                    wav_filepath, feature_params)

    def get_feature(self, channel):
        """
        Args:
            channel (str): channel id ('L' or 'R')
        Returns:
            numpy.ndarray: shape=(number of frames, filter bank)
                feature matrix of the channel
        """
        if self.is_monoral:
            assert channel == 'L'
        assert channel in ('L', 'R')
        assert self._feature[channel] is not None
        return self._feature[channel]


class CSJParser(object):

    """
    Parameters:
        base_dir (str): root path of CSJ dataset
        lexicon (asr.decoder.Lexicon): Lexicon object
    """

    def __init__(self, base_dir, lexicon):
        self.base_dir = base_dir
        self.xml_dir = os.path.join(base_dir, 'XML/BaseXML')
        self.wav_dir = os.path.join(base_dir, 'WAV')
        self.lexicon = lexicon

    def parse(self, feature_params):
        """
        Args
            feature_params (asr.feature.FeatureParams): FeatureParams object
        Returns:
            list[numpy.ndarray]:
                list of array whose shape is (number of frames, feature size)
            list[numpy.ndarray]:
                list of array whose shape is (number of labels,)
        """
        X = []
        Y = []
        file_list_path = os.path.join(self.base_dir, 'fileList.csv')
        with open(file_list_path, encoding='shift_jis') as f:
            reader = csv.DictReader(f)
            for row in reader:
                talk_id = row['講演ID']
                talk_category = row['講演種別']
                core_or_noncore = row['コア']
                csj_talk = CSJTalk(talk_id, talk_category, core_or_noncore)
                if csj_talk.is_core is False:
                    print('Skip {}'.format(csj_talk.id))
                    continue
                x, y = self.parse_one(csj_talk, feature_params)
                X.extend(x)
                Y.extend(y)
        return X, Y

    def parse_one(self, csj_talk, feature_params):
        """Process one CSJ talk
        Args:
            csj_talk (asr.dataset.CSJTalk): CSJTalk object
            feature_params (asr.feature.FeatureParams): FeatureParams object
        Returns:
            list[numpy.ndarray]:
                list of array whose shape is (number of frames, feature size)
            list[numpy.ndarray]:
                list of array whose shape is (number of labels,)
        """
        print('Process {} ...'.format(csj_talk.id))
        csj_talk.extract_feature(self.wav_dir, feature_params)
        csj_talk.load_xml(self.xml_dir)
        X = []
        labels = []
        for ipu in csj_talk.get_xml().getElementsByTagName('IPU'):
            ipu_id = ipu.getAttribute('IPUID')
            label = self.create_label(ipu)
            if label.shape[0] == 0:
                print('Skip IPU because label length is 0: IPUID={}'.format(
                    ipu_id))
                continue
            data = self.create_data(ipu, csj_talk, feature_params.hop_length)
            X.append(data)
            labels.append(label)
        return X, labels

    def create_data(self, ipu, csj_talk, hop_length_in_sec):
        """Create feature matrix for one IPU
        Args:
            ipu (xml.dom.minidom.Element): IPU tag
            csj_talk (asr.dataset.CSJTalk): CSJTalk object
            hop_length_in_sec (float): frame interval in seconds
        Returns:
            numpy.ndarray: shape=(number of frames, feature size)
        """
        start_time = float(ipu.getAttribute('IPUStartTime'))
        end_time = float(ipu.getAttribute('IPUEndTime'))
        channel = ipu.getAttribute('Channel')
        audio_feature = csj_talk.get_feature(channel)
        start_frame_idx = int(start_time / hop_length_in_sec)
        end_frame_idx = int(end_time / hop_length_in_sec)
        return audio_feature[start_frame_idx:end_frame_idx]

    def create_label(self, ipu):
        """Create labels for one IPU
        Args:
            ipu (xml.dom.minidom.Element): IPU tag
        Returns:
            numpy.ndarray: shape=(number of labels,)
                1D array of label id
        """
        labels = []
        ipu_id = ipu.getAttribute('IPUID')
        for luw in ipu.getElementsByTagName('LUW'):
            luw_id = luw.getAttribute('LUWID')
            for suw in luw.getElementsByTagName('SUW'):
                suw_id = suw.getAttribute('SUWID')
                if self.is_masked(suw):
                    print('Skip IPU because of it contains masked SUW: '
                          'IPUID={}'.format(ipu_id))
                    return numpy.array([])
                word = suw.getAttribute('PlainOrthographicTranscription')
                if word == '':
                    print(
                        "Skip SUW because PlainOrthographicTranscription='': "
                        'IPUID={}, LUWID={}, SUWID={}'.format(
                            ipu_id, luw_id, suw_id))
                    continue
                reading = []
                for mora in suw.getElementsByTagName('Mora'):
                    mora_id = mora.getAttribute('MoraID')
                    kana = mora.getAttribute('MoraEntity')
                    if kana in ('', 'φ',):
                        print("Skip Mora because MoraEntity='{}': "
                              'IPUID={}, LUWID={}, SUWID={}, MoraID={}'.format(
                                kana, ipu_id, luw_id, suw_id, mora_id))
                        continue
                    reading.append(kana)
                self.lexicon.add(word, reading)
                labels.extend(self.lexicon.get(word))
        return numpy.array(labels, dtype=numpy.int32)

    def is_masked(self, suw):
        trans_suw = suw.getElementsByTagName('TransSUW')[0]
        if trans_suw.getAttribute('TagMaskStart') == '1':
            return True
        elif trans_suw.getAttribute('TagMaskMidst') == '1':
            return True
        elif trans_suw.getAttribute('TagMaskEnd') == '1':
            return True
        else:
            return False


class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert isinstance(self.data[idx], torch.Tensor)
        assert isinstance(self.labels[idx], torch.Tensor)
        return self.data[idx], self.labels[idx]

    def save(self, path):
        # NOTE: Save data and labels as numpy.ndarray
        #       because saving them as torch.Tensor requires a lot of memory
        obj = {'data': self.data, 'labels': self.labels}
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return AudioDataset(obj['data'], obj['labels'])

    def to_torch(self):
        self.data = [torch.from_numpy(x) for x in self.data]
        self.labels = [torch.from_numpy(y) for y in self.labels]
