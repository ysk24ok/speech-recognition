import argparse
import os

from asr.dataset.csj import CSJParser
from asr.decoder import Corpus
from asr.label_table import VocabularyTable


parser = argparse.ArgumentParser()
parser.add_argument('workdir', type=str,
                    help='Directory path where files are loaded and saved')
parser.add_argument('dataset_path', type=str, help='Dataset path')
parser.add_argument('--corpus-file', type=str, default='corpus.txt',
                    help='Corpus file name')
parser.add_argument('--vocabulary-table-file', type=str, default='vocab.txt',
                    help='Vocabulary table file name')
parser.add_argument('--dataset-type', type=str, default='csj', choices=['csj'],
                    help='Dataset type to use')
parser.add_argument('--use-subset', action='store_true',
                    help='A flag whether to use a subset of the dataset')
args = parser.parse_args()

print('Creating vocabulary table and corpus ...')
corpus = Corpus()
vocabulary_table = VocabularyTable()
if args.dataset_type == 'csj':
    csj_parser = CSJParser(args.dataset_path)
    tr_talk_sets, _ = csj_parser.get_talks(1)
    for tr_talks in tr_talk_sets:
        csj_parser.add_vocabulary(tr_talks, vocabulary_table, corpus,
                                  only_core=args.use_subset)
else:
    raise ValueError('dataset_type: {} is not supported'.format(
        args.dataset_type))
print('Saving corpus ...')
corpus_path = os.path.join(args.workdir, args.corpus_file)
corpus.save(corpus_path)
print('Saving vocabulary table ...')
vocabulary_table_path = os.path.join(args.workdir, args.vocabulary_table_file)
vocabulary_table.save(vocabulary_table_path)
