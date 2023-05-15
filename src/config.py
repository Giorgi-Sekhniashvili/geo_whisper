from pydantic import BaseSettings, Field


class Config(BaseSettings):
    prepare_dataset: bool
    save_to_disk: bool

    dataset_name: str
    dataset_lang: str
    model_name: str
    model_lang: str
    task: str
    output_model_name: str

    # dataset arguments
    sample_percentage: float

    sampling_rate: int
    do_lower_case: bool
    do_remove_punctuation: bool

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class TrainingArgumentsConfig(BaseSettings):
    output_dir: str = Field(..., env="OUTPUT_MODEL_NAME")

    # training arguments
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    fp16: bool
    evaluation_strategy: str
    predict_with_generate: bool
    generation_max_length: int
    weight_decay: float
    save_steps: int
    eval_steps: int
    logging_steps: int
    report_to: list
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    push_to_hub: bool

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
