from pydantic import BaseSettings


class Config(BaseSettings):
    prepare_dataset: bool = False

    dataset_name: str = "mozilla-foundation/common_voice_13"
    dataset_lang: str = "ka"
    model_name: str = "GiorgiSekhniashvili/whisper-tiny-ka"
    model_lang: str = "Georgian"
    task: str = "transcribe"
    output_model_name: str = "whisper-tiny-ka"

    sampling_rate: int = 16_000
    do_lower_case: bool = False
    do_remove_punctuation: bool = False

    # training arguments
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 5000
    gradient_checkpointing: bool = True
    fp16: bool = True
    evaluation_strategy: str = "steps"
    per_device_eval_batch_size: int = 8
    predict_with_generate: bool = True
    generation_max_length: int = 256
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 25
    report_to: list = ["tensorboard"]
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    push_to_hub: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
