#!/usr/bin/env python3
import logging
import sys
import time
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from datasets import DatasetDict, load_dataset
from tqdm import tqdm

import flax
import jax
import jax.numpy as jnp
import torchaudio
import optax
from flax import jax_utils, traverse_util
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from transformers import (
    FlaxWav2Vec2ForPreTraining,
    HfArgumentParser,
    TrainingArguments,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    is_tensorboard_available,
)
from transformers.models.wav2vec2.modeling_flax_wav2vec2 import _compute_mask_indices, _sample_negative_indices


logger = logging.getLogger(__name__)


@flax.struct.dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    max_gumbel_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Maximum temperature for gumbel softmax."}
    )
    min_gumbel_temperature: Optional[float] = field(
        default=0.1, metadata={"help": "Minimum temperature for gumbel softmax."}
    )
    gumbel_temperature_decay: Optional[float] = field(
        default=0.999995, metadata={"help": "Decay of gumbel temperature during training."}
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@flax.struct.dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    validation_split_name: Optional[str] = field(
        default="validation",
        metadata={
            "help": "The name of the validation data set split to use (via the datasets library). Defaults to 'validation'"
        },
    )
    speech_file_column: Optional[str] = field(
        default="file",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'file'"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_duration_in_seconds: Optional[float] = field(
        default=20.0, metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds"}
    )
    pad_to_multiple_of: Optional[int] = field(
        default=1024,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is important to avoid triggering recompilations on TPU"
        },
    )


def configure_logger(model_args: ModelArguments, training_args: TrainingArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if model_args.verbose_logging:
        logging_level = logging.DEBUG
    logger.setLevel(logging_level)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    configure_logger(model_args, training_args)

    # Downloading and loading a dataset from the hub.
    datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)

    if "validation" not in datasets.keys():
        # make sure only "validation" and "train" keys remain"
        datasets = DatasetDict()
        datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
        )
        datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
        )
    else:
        # make sure only "validation" and "train" keys remain"
        datasets = DatasetDict()
        datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split="validation",
            cache_dir=model_args.cache_dir,
        )
        datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"{data_args.train_split_name}",
            cache_dir=model_args.cache_dir,
        )

    # only normalized-inputs-training is supported
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir, do_normalize=True
    )

    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    # Preprocessing the datasets.
    # We need to read the aduio files as arrays and tokenize the targets.
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = torchaudio.load(batch["path"])
        batch["speech"] = resampler(speech_array).squeeze().numpy()
        return batch

    # load audio files into numpy arrays
    vectorized_datasets = datasets.map(
        speech_file_to_array_fn, num_proc=data_args.preprocessing_num_workers, remove_columns=datasets["train"].column_names,
    )

    # filter audio files that are too long
    vectorized_datasets = vectorized_datasets.filter(
        lambda data: len(data["speech"]) < int(data_args.max_duration_in_seconds * feature_extractor.sampling_rate),
        keep_in_memory=True,
    )

    def normalize(batch):
        return feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate)

    # normalize and transform to `BatchFeatures`
    vectorized_datasets = vectorized_datasets.map(
        normalize,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        remove_columns=vectorized_datasets["train"].column_names,
    )

    # save vectorized dataset once
    vectorized_datasets.save_to_disk(model_args.cache_dir)

if __name__ == "__main__":
    main()
