# Modifications of the setup script

Contains the issues that lead to the modification of the original setup and training script.

### Issue

Dependency `librosa` is not installed.

```
ModuleNotFoundError: No module named 'librosa'
```

#### Workaround

```
pip install librosa
```

### Issue

```
OSError: sndfile library not found
```

#### Workaround

```
sudo apt install libsndfile1
```

### Issue

```
ValueError: Unknown split "train.100". Should be one of ['train', 'test', 'validation', 'other', 'invalidated'].
```

#### Solution

Update the training script changing
```
--train_split_name="train.100" \
```
for
```
--validation_split_percentage="5" \
```

### Issue

```
  File "./run_wav2vec2_pretrain_flax.py", line 315, in prepare_dataset
    batch["speech"], _ = librosa.load(batch[data_args.speech_file_column], sr=feature_extractor.sampling_rate)
KeyError: 'file'
```

#### Solution

Specify the speech file column in the training script by adding:

```
--speech_file_column="path" \
```

### Issue

```
RuntimeError: Error opening '/home/mariagrandury/.cache/huggingface/datasets/downloads/extracted/bd58f2e7808a2802cb11d9aae2673fa0a1e54b008404f75a1c63c2751332b806/cv-corpus-6.1-2020-12-11/es/clips/common_voice_es_19819446.mp3': File contains data in an unknown format.
```

#### Solution

Specify the format by adding:

```
--dtype="bfloat16" \
```

### Issue

Warning "PySoundFile failed. Trying audioread instead." is spaming the terminal (https://github.com/librosa/librosa/issues/1015).

#### Workaround

Ignore warnings with:

```
python -W ignore
```