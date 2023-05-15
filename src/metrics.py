import evaluate

metric = evaluate.load("wer")

# evaluate with the 'normalised' WER
do_normalize_eval = False


def compute_metrics(pred, processor):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # if do_normalize_eval:
    #     pred_str = [normalizer(pred) for pred in pred_str]
    #     label_str = [normalizer(label) for label in label_str]
    #     # filtering step to only evaluate the samples that correspond to non-zero references:
    #     pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    #     label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
