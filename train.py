import logging
from functools import partial

from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.callbacks import ShuffleCallback
from src.config import Config, TrainingArgumentsConfig
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_metrics
from src.prepare_dataset import prepare_dataset

logging.basicConfig(level=logging.INFO)


def train():
    config = Config()
    training_args_config = TrainingArgumentsConfig()
    training_args = Seq2SeqTrainingArguments(**training_args_config.dict())

    if config.prepare_dataset:
        dataset, _ = prepare_dataset(config)
    else:
        dataset = load_dataset(config.dataset_name, config.dataset_lang)
    logging.info("Training model...")

    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    processor = WhisperProcessor.from_pretrained(
        config.model_name, task=config.task, language=config.model_lang
    )
    compute_metrics_fn = partial(compute_metrics, processor=processor)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics_fn,
        tokenizer=processor,
        callbacks=[ShuffleCallback()],
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
