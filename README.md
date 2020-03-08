# Speech Recognition

PyTorch implementation of end-to-end neural automatic speech recognition.

Following papers are implemented:

- [EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding](http://www.cs.cmu.edu/~fmetze/interACT/Publications_files/publications/eesenasru.pdf)
- [Listen, Attend and Spell](https://research.google/pubs/pub44926.pdf)

Supported dataset is currently [CSJ: Corpus of Spontaneous Japanese](https://pj.ninjal.ac.jp/corpus_center/csj/) only ...

## scripts usage

You have to pass `workdir` argument to all scripts so you have to create it in advance.
The scripts create files under `workdir` directory. You can override the file name by specifying corresponding commandline options.
In addition, you have to run scripts with `PYTHONPATH=.` to add the current directory to `sys.path` .

## creating vocabulary

`scripts/create_vocabulary_table.py` creates a vocabulary table and a corpus for a language model.

```console
$ PYTHONPATH=. python3 scripts/create_vocabulary_table.py /path/to/work_directory /path/to/dataset
```

By default, a vocabulary table is created at `/path/to/work_directory/vocab.txt` and a corpus file at `/path/to/work_directory/corpus.txt` .

You can add `--use-subset` option to create a vocabulary table and a corpus from a subset of the dataset.

## creating training data

Use `scripts/create_training_data.py` to create training/development data. You can pass options such as `--feature-size` for feature extraction.

```console
$ PYTHONPATH=. python3 scripts/create_training_data.py /path/to/work_directory /path/to/dataset
```

By default, training data is created under `/path/to/work_directory/trdir` directory and development data under `/path/to/work_directory/devdir` directory.
Training data is splited into multiple files and you can change the number of files by `--training-data-file-count` option. Pass a larger number to the option to reduce memory consumption.

You can add `--use-subset` option if you want to train your model with smaller data.

For EESEN, you have to pass `--label-type phoneme` and `--create-lexicon` .

```console
$ PYTHONPATH=. python3 scripts/create_training_data.py /path/to/work_directory /path/to/dataset \
  --label-type phoneme --create-lexicon
```

## Listen Attend Spell (LAS)

### training

Run `scripts/las/train.py` to train LAS model.
You can pass several model parameter options. Run with `--help` option to see more details.

```console
$ PYTHONPATH=. python3 scripts/las/train.py /path/to/work_directory
```

By default, a model file is created at `/path/to/work_directory/model.bin` .

You can load a saved model file to resume training by adding `--resume` option.

### recognizing audio files

TODO


## EESEN

### creating WFST decoder

Run `scripts/eesen/create_decoder.py` to create a WFST decoder.

```console
$ PYTHONPATH=. python3 scripts/eesen/create_decoder.py /path/to/work_directory
```

By default, a WFST decoder file is created at `/path/to/work_directory/decoder.fst` .

### training

Run `scripts/eesen/train.py` to train EESEN model.
You can pass several model parameter options. Run with `--help` option to see more details.

```console
$ PYTHONPATH=. python3 scripts/eesen/train.py /path/to/work_directory
```

By default, a model file is created at `/path/to/work_directory/model.bin` .

You can load a saved model file to resume training by adding `--resume` option.

### recognizing audio files

After training, you can recognize audio files by running `scripts/eesen/recognize.py` .

```console
$ PYTHONPATH=. python3 scripts/eesen/recognize.py /path/to/work_directory \
  /path/to/audio_file1 /path/to/audio_file2 ...
```
