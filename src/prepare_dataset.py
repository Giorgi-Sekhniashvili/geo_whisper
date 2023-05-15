import logging
from pathlib import Path

from datasets import Audio, load_dataset
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from src.config import Config
from datasets import DatasetDict


def preprocess(
    batch,
    processor,
    do_lower_case,
    do_remove_punctuation,
    normalizer,
    sampling_rate=16_000,
):
    # load and (possibly) resample audio data to 16kHz
    audio_list = [audio for audio in batch["audio"]]
    array_list = [audio["array"] for audio in audio_list]

    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(
        array_list, sampling_rate=sampling_rate
    ).input_features

    logging.info(f'{batch["input_features"][0].shape = }')

    # compute input length of audio sample in seconds than used to filter out samples longer than 30 seconds
    batch["input_length"] = [
        len(audio["array"]) / audio["sampling_rate"] for audio in audio_list
    ]
    logging.info(f'{batch["input_length"] = }')
    # optional pre-processing steps
    transcription = batch["sentence"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()

    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


def prepare_dataset(config: Config) -> tuple[DatasetDict, Path]:
    dataset = load_dataset(config.dataset_name, config.dataset_lang)
    logging.info(f"Dataset loaded: {dataset}")
    dataset = DatasetDict(
        {
            "train": dataset["train"].select(range(10)),
            "validation": dataset["validation"].select(range(10)),
            "test": dataset["test"].select(range(10)),
        }
    )

    logging.info(f"Dataset loaded: {dataset}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.sampling_rate))

    processor = WhisperProcessor.from_pretrained(
        config.model_name, task=config.task, language=config.model_lang
    )

    normalizer = BasicTextNormalizer()

    vectorized_dataset = dataset.map(
        preprocess,
        remove_columns=list(
            next(iter(dataset.values())).features,
        ),
        batched=True,
        fn_kwargs={
            "processor": processor,
            "do_lower_case": config.do_lower_case,
            "do_remove_punctuation": config.do_remove_punctuation,
            "normalizer": normalizer,
        },
    ).with_format("torch")

    save_path = Path(f"data/{config.dataset_name}")
    save_path.mkdir(parents=True, exist_ok=True)
    vectorized_dataset.save_to_disk(save_path)
    return vectorized_dataset, save_path
