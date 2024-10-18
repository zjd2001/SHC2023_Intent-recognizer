import numpy as np

from paddlenlp.utils.log import logger

# 预处理
def preprocess_function(examples, tokenizer, max_length, is_test=False):
    """
    Builds model inputs from a sequence for sequence classification tasks
    by concatenating and adding special tokens.
    """
    result = tokenizer(examples["text"], max_length=max_length, truncation=True)
    if not is_test:
        result["labels"] = np.array([examples["label"]], dtype="int64")
    return result

# 读取数据集
def read_local_dataset(path, label2id=None, is_test=False):
    """
    Read dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if is_test:
                sentence = line.strip()
                yield {"text": sentence}
            else:
                items = line.strip().split("\t")
                yield {"text": items[0], "label": label2id[items[1]]}


# 打印日志                
def log_metrics_debug(output, id2label, dev_ds, bad_case_path):
    """
    Log metrics in debug mode.
    """
    predictions, label_ids, metrics = output
    pred_ids = np.argmax(predictions, axis=-1)
    logger.info("-----Evaluate model-------")
    logger.info("Dev dataset size: {}".format(len(dev_ds)))
    logger.info("Accuracy in dev dataset: {:.2f}%".format(metrics["test_accuracy"] * 100))
    logger.info(
        "Macro average | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
            metrics["test_macro avg"]["precision"] * 100,
            metrics["test_macro avg"]["recall"] * 100,
            metrics["test_macro avg"]["f1-score"] * 100,
        )
    )
    for i in id2label:
        l = id2label[i]
        logger.info("Class name: {}".format(l))
        i = "test_" + str(i)
        if i in metrics:
            logger.info(
                "Evaluation examples in dev dataset: {}({:.1f}%) | precision: {:.2f} | recall: {:.2f} | F1 score {:.2f}".format(
                    metrics[i]["support"],
                    100 * metrics[i]["support"] / len(dev_ds),
                    metrics[i]["precision"] * 100,
                    metrics[i]["recall"] * 100,
                    metrics[i]["f1-score"] * 100,
                )
            )
        else:
            logger.info("Evaluation examples in dev dataset: 0 (0%)")
        logger.info("----------------------------")

    with open(bad_case_path, "w", encoding="utf-8") as f:
        f.write("Text\tLabel\tPrediction\n")
        for i, (p, l) in enumerate(zip(pred_ids, label_ids)):
            p, l = int(p), int(l)
            if p != l:
                f.write(dev_ds.data[i]["text"] + "\t" + id2label[l] + "\t" + id2label[p] + "\n")

    logger.info("Bad case in dev dataset saved in {}".format(bad_case_path))
