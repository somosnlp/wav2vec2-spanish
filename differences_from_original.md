### Issue

Requirement librosa not installed

###### Error Message

```
ModuleNotFoundError: No module named 'librosa'
```

###### Workaround

pip install librosa

### Issue

```
OSError: sndfile library not found
```

###### Workaround

```
sudo apt install libsndfile1
```

### Issue

```
ValueError: Unknown split "train.100". Should be one of ['train', 'test', 'validation', 'other', 'invalidated'].
```

##### Solution

Change
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

##### Solution
add 
```
--speech_file_column="path" \
```

### Issue
```
RuntimeError: Error opening '/home/mariagrandury/.cache/huggingface/datasets/downloads/extracted/bd58f2e7808a2802cb11d9aae2673fa0a1e54b008404f75a1c63c2751332b806/cv-corpus-6.1-2020-12-11/es/clips/common_voice_es_19819446.mp3': File contains data in an unknown format.
```

##### Solution
Add

```
--dtype="bfloat16" \
```
