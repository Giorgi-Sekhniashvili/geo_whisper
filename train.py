import logging

from datasets import load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.callbacks import ShuffleCallback
from src.config import Config
from src.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.metrics import compute_metrics
from src.prepare_dataset import prepare_dataset

logging.basicConfig(level=logging.INFO)


def train():
    config = Config()
    if config.prepare_dataset:
        dataset, dataset_path = prepare_dataset(config)
    else:
        dataset = load_dataset(config.dataset_name, config.dataset_lang)
    print("Training model...")

    model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
    processor = WhisperProcessor.from_pretrained(
        config.model_name, task=config.task, language=config.model_lang
    )
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_model_name,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        fp16=config.fp16,
        evaluation_strategy=config.evaluation_strategy,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        predict_with_generate=config.predict_with_generate,
        generation_max_length=config.generation_max_length,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=config.report_to,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        push_to_hub=config.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[ShuffleCallback()],
    )

    trainer.train()
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
