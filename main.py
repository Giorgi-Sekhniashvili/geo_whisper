import librosa
from transformers import WhisperForConditionalGeneration, AutoProcessor


if __name__ == "__main__":
    audio_path = "data/Recording.m4a"
    model_name = "GiorgiSekhniashvili/whisper-tiny-ka"

    processor = AutoProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="Georgian", task="transcribe"
    )

    waveform, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(waveform, sampling_rate=sr, return_tensors="pt")
    print(f"{input_values.keys() = }")
    res = model.generate(
        input_values["input_features"],
        forced_decoder_ids=forced_decoder_ids,
        max_new_tokens=448,
    )
    print(processor.batch_decode(res, skip_special_tokens=True))
    print("Done!")
