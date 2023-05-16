import logging
from pathlib import Path

from datasets import Audio, DatasetDict, load_dataset
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from src.config import Config


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

    # compute input length of audio sample in seconds than used to filter out samples longer than 30 seconds
    batch["input_length"] = [
        len(audio["array"]) / audio["sampling_rate"] for audio in audio_list
    ]
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

    train_sample_count = max(int(len(dataset["train"]) * config.sample_percentage), 10)
    validation_sample_count = max(
        int(len(dataset["validation"]) * config.sample_percentage), 10
    )
    dataset = DatasetDict(
        {
            "train": dataset["train"].select(range(train_sample_count)),
            "validation": dataset["validation"].select(range(validation_sample_count)),
            "test": dataset["test"],
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
        batch_size=1024,
        fn_kwargs={
            "processor": processor,
            "do_lower_case": config.do_lower_case,
            "do_remove_punctuation": config.do_remove_punctuation,
            "normalizer": normalizer,
        },
    ).with_format("torch")

    if config.save_to_disk:
        save_path = Path(f"data/{config.dataset_name}")
        save_path.mkdir(parents=True, exist_ok=True)
        vectorized_dataset.save_to_disk(save_path)
    else:
        save_path = None
    return vectorized_dataset, save_path
