import csv
import os
import xml.dom.minidom

import numpy

from . import DatasetParser
from ..feature import extract_feature_from_wavfile


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

    def unref(self):
        self._feature['L'] = None
        self._feature['R'] = None
        self._xml_content = None


class CSJParser(DatasetParser):

    """
    Parameters:
        base_dir (str): root path of CSJ dataset
        corpus (asr.decoder.Corpus): Corpus object
        lexicon (asr.decoder.Lexicon): Lexicon object
    """

    # See chapter 5 in https://pj.ninjal.ac.jp/corpus_center/csj/manu-f/asr.pdf
    talk_ids_for_dev_sets = [
        [
            'A01M0097', 'A01M0110', 'A01M0137', 'A03M0106', 'A03M0112',
            'A03M0156', 'A04M0051', 'A04M0121', 'A04M0123', 'A05M0011'
        ],
        [
            'A01M0056', 'A01M0141', 'A02M0012', 'A03M0016', 'A06M0064',
            'A01F0001', 'A01F0034', 'A01F0063', 'A03F0072', 'A06F0135'
        ],
        [
            'S00M0008', 'S00M0070', 'S00M0079', 'S00M0112', 'S00M0213',
            'S00F0019', 'S00F0066', 'S00F0148', 'S00F0152', 'S01F0105'
        ]
    ]

    def __init__(self, base_dir, corpus, lexicon):
        self.base_dir = base_dir
        self.xml_dir = os.path.join(base_dir, 'XML/BaseXML')
        self.wav_dir = os.path.join(base_dir, 'WAV')
        self.corpus = corpus
        self.lexicon = lexicon

    def get_talks(self, training_data_file_count):
        """
        Args:
            training_data_file_count (int): Training file counts to be saved
        Returns:
            list[list[asr.dataset.CSJTalk]]: training sets
            list[list[asr.dataset.CSJTalk]]: development sets
        """
        file_list_path = os.path.join(self.base_dir, 'fileList.csv')
        tr_talks = [[] for _ in range(training_data_file_count)]
        dev_talks = [[] for _ in range(len(self.talk_ids_for_dev_sets))]
        file_count = 0
        with open(file_list_path, encoding='shift_jis') as f:
            reader = csv.DictReader(f)
            for row in reader:
                talk_id = row['講演ID']
                talk_category = row['講演種別']
                core_or_noncore = row['コア']
                csj_talk = CSJTalk(talk_id, talk_category, core_or_noncore)
                # dev set
                dev_set_id = self._get_dev_set_id(talk_id)
                if dev_set_id is not None:
                    dev_talks[dev_set_id].append(csj_talk)
                # tr set
                idx = file_count % training_data_file_count
                tr_talks[idx].append(csj_talk)
                file_count += 1
        return tr_talks, dev_talks

    def _get_dev_set_id(self, talk_id):
        for i, talk_ids in enumerate(self.talk_ids_for_dev_sets):
            if talk_id in talk_ids:
                return i
        # This talk_id is for training set
        return None

    def parse(self, csj_talks, feature_params, only_core=False):
        """
        Args
            csj_talks (list[asr.dataset.CSJTalk]): CSJTalk objects
            feature_params (asr.feature.FeatureParams): FeatureParams object
            only_core (bool): A flag which indicates parsing only core files
        Returns:
            list[numpy.ndarray]:
                list of array whose shape is (number of frames, feature size)
            list[numpy.ndarray]:
                list of array whose shape is (number of labels,)
        """
        data, labels = [], []
        for csj_talk in csj_talks:
            if only_core is True and csj_talk.is_core is False:
                print('Skip noncore {}'.format(csj_talk.id))
                continue
            d, label = self.parse_one(csj_talk, feature_params)
            csj_talk.unref()
            data.extend(d)
            labels.extend(label)
        return data, labels

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
        data = []
        labels = []
        for ipu in csj_talk.get_xml().getElementsByTagName('IPU'):
            ipu_id = ipu.getAttribute('IPUID')
            label = self.create_label(ipu)
            if label.shape[0] == 0:
                print('Skip IPU because label length is 0: IPUID={}'.format(
                    ipu_id))
                continue
            d = self.create_data(ipu, csj_talk, feature_params.hop_length)
            data.append(d)
            labels.append(label)
        return data, labels

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
                end_of_sentence = False
                if suw.getAttribute('ClauseBoundaryLabel') == '[文末]':
                    end_of_sentence = True
                self.corpus.add(word, end_of_sentence=end_of_sentence)
                moras = []
                for mora in suw.getElementsByTagName('Mora'):
                    mora_id = mora.getAttribute('MoraID')
                    kana = mora.getAttribute('MoraEntity')
                    if kana in ('', 'φ',):
                        print("Skip Mora because MoraEntity='{}': "
                              'IPUID={}, LUWID={}, SUWID={}, MoraID={}'.format(
                                kana, ipu_id, luw_id, suw_id, mora_id))
                        continue
                    moras.append(kana)
                self.lexicon.add(word, moras)
                labels.extend(self.lexicon.get_label_ids(moras))
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
