import json
import re
from typing import Callable, Optional

import torch
from datasets import DatasetDict, load_dataset
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

INSTRUCTION = """
Give your answer as a JSON dictionary with the "option" (a letter from A-E) and the  corresponding"option_text". No yapping.
""".strip()


def load_medqa_data() -> DatasetDict:
    """
    dataset["train"][i]

    # input
    Q:A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient??
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'},

    # instruction
    Please answer with one of the option in the bracket

    # output
    E: Lipohyalinosis
    """
    dataset = load_dataset("medalpaca/medical_meadow_medqa")
    assert isinstance(dataset, DatasetDict)
    return dataset


def load_train_test_data(
    train_size: int,
    test_size: int,
    input_limit: Optional[int] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> DatasetDict:
    """
    train_set[i]

    # input
    Q: A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient?? Give your answer as a JSON dictionary in the form of {"option": "A-E", "text": "corresponding text"}. No yapping.
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'}

    # output
    {"option": "E", "text": "Lipohyalinosis"}

    # true_label
    E
    """
    dataset = load_medqa_data()
    if isinstance(input_limit, int):
        dataset = dataset.filter(lambda sample: len(sample["input"]) <= input_limit)
    dataset = dataset["train"].train_test_split(
        train_size=train_size,
        test_size=test_size,
        shuffle=shuffle,
        seed=seed,
    )
    dataset = dataset.remove_columns("instruction")
    return dataset


def reformat_sample(sample: dict[str, str]) -> dict[str, str]:
    input = "Q: " + sample["input"].removeprefix("Q:").removesuffix(",")
    input += "\n" + INSTRUCTION
    answer_option = sample["output"][0]
    answer_text = sample["output"][3:]
    true_label = answer_option
    output = json.dumps({"option": answer_option, "text": answer_text})
    return {"input": input, "output": output, "true_label": true_label}


def get_generated_texts(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    output_ids: torch.Tensor,
    remove_eos: bool = True,
) -> list[str]:
    """Retreive only the generated texts based on input and output ids

    Args:
        input_ids (torch.Tensor): batch of input ids.
        output_ids (torch.Tensor): corresponding output ids.
        remove_eos (bool, optional): whether to remove the final <eos> token. Defaults to True.

    Returns:
        list[str]: the batch of text generated by the model based on the input texts.

    Example:
    # input text
    <bos><start_of_turn>user
    Q: A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient?? Give your answer as a JSON dictionary in the form of {"option": "A-E", "text": "corresponding text"}. No yapping.
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'}<end_of_turn>
    <start_of_turn>model

    # output text
    <bos><start_of_turn>user
    Q: A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient?? Give your answer as a JSON dictionary in the form of {"option": "A-E", "text": "corresponding text"}. No yapping.
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'}<end_of_turn>
    <start_of_turn>model
    {"option": "A-Berry aneurysm rupture", "text": "Berry aneurysm rupture"}<eos>

    # returns (with remove_eos = True)
    {"option": "A-Berry aneurysm rupture", "text": "Berry aneurysm rupture"}
    """
    texts = [
        tokenizer.decode(out_seq[len(in_seq) :])
        for in_seq, out_seq in zip(input_ids, output_ids)
    ]
    if remove_eos:
        texts = [text.removesuffix("<eos>") for text in texts]
    return texts


def get_available_qa_choices(passage: str) -> dict[str, str]:
    """
    # passage
    Q:A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient??
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'},

    # returns
    {
        'A': 'Early disseminated Lyme disease',
        'B': 'Embolic stroke at the posterior inferior cerebellar artery (PICA)',
        'C': 'Hypoperfusion of the anterior spinal artery (ASA)',
        'D': 'Labryrinthitis',
        'E': 'Thrombotic stroke at the anterior inferior cerebellar artery (AICA)'
    }
    """
    return {
        match.group(1): match.group(2)
        for match in re.finditer("'([^']+)': '([^']+)'", passage)
    }


def get_best_match(answer_text: str, passage: str) -> str:
    """
    # passage
    Q:A 65-year-old man presents to the emergency department for sudden weakness. The patient states that he was at home enjoying his morning coffee when his symptoms began. He says that his left arm suddenly felt very odd and weak thus prompting him to come to the ED. The patient has a past medical history of diabetes, COPD, hypertension, anxiety, alcohol abuse, and PTSD. He recently fell off a horse while horseback riding but claims to not have experienced any significant injuries. He typically drinks 5-7 drinks per day and his last drink was yesterday afternoon. His current medications include insulin, metformin, atorvastatin, lisinopril, albuterol, and fluoxetine. His temperature is 99.5°F (37.5°C), blood pressure is 177/118 mmHg, pulse is 120/min, respirations are 18/min, and oxygen saturation is 93% on room air. On physical exam, you note an elderly man who is mildly confused. Cardiopulmonary exam demonstrates bilateral expiratory wheezes and a systolic murmur along the right upper sternal border that radiates to the carotids. Neurological exam reveals cranial nerves II-XII as grossly intact with finger-nose exam mildly abnormal on the left and heel-shin exam within normal limits. The patient has 5/5 strength in his right arm and 3/5 strength in his left arm. The patient struggles to manipulate objects such as a pen with his left hand. The patient is given a dose of diazepam and started on IV fluids. Which of the following is the most likely diagnosis in this patient??
    {'A': 'Berry aneurysm rupture', 'B': 'Bridging vein tear', 'C': 'Cerebellar bleeding', 'D': 'Hypertensive encephalopathy', 'E': 'Lipohyalinosis'},

    # answer_text
    Berry aneurysm rupture

    # returns
    A
    """
    choices = get_available_qa_choices(passage)
    if not choices:
        return ""  # no choices found
    return sorted(
        choices,
        key=lambda c: fuzz.token_set_ratio(
            choices[c], answer_text, processor=default_process
        ),
    )[-1]


def parse_prediction(pred_text: str, passage: str = "") -> str:
    """Parse the predicted answer based on the output text, with text matching as backup.

    Args:
        pred_text (str): text outputted from the language model.
        passage (str, optional): The input passage for text matchingin case the LLM does
            not output parseable JSON. Useful for evaluating the model before finetuning.
            Defaults to "" (no passage); in this case the backup prediction will be "".

    Returns:
        str: option (a letter or "") predicted by the model.

    Example 1: no passage; returns "A"
    {
    "option": "A",
    "answer": "Early disseminated Lyme disease"
    }<eos>

    Example 2: no passage; returns ""
    {"option": "A - Early disseminated Lyme disease"}

    Example 3: with passage; returns "A"
    {"option": "A - Early disseminated Lyme disease"}
    """

    json_text = pred_text
    if match := re.search(r"\{", json_text):  # remove anything before first {
        json_text = json_text[match.start() :]
    if match := re.search(r"\}", json_text[::-1]):  # remove anything after last }
        json_text = json_text[: len(json_text) - match.start()]

    try:
        if match := re.match(r"^[a-eA-E]$", json.loads(json_text)["option"].strip()):
            return match.group(0)
    except (json.JSONDecodeError, KeyError):
        pass

    # backup: if a passage is supplied, get the best match
    if passage:
        return get_best_match(pred_text, passage)
    else:
        return ""


def create_predict(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    model: PreTrainedModel,
    model_alias: str,
    batch: bool = False,
    generate_kwargs: Optional[dict] = None,
) -> Callable[[dict], dict]:

    generate_kwargs = generate_kwargs or {}

    def _batch_predict(samples: dict) -> dict:
        input_ids = tokenizer.apply_chat_template(
            [[{"role": "user", "content": text}] for text in samples["input"]],
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(input_ids, **generate_kwargs)
        pred_texts = get_generated_texts(tokenizer, input_ids, output_ids)
        samples[f"{model_alias}_pred"] = pred_texts
        samples[f"{model_alias}_label"] = [
            parse_prediction(pred, text)
            for text, pred in zip(samples["input"], pred_texts)
        ]
        return samples

    def _predict(sample: dict) -> dict:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["input"]}],
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(input_ids, **generate_kwargs)
        pred_text = get_generated_texts(tokenizer, input_ids, output_ids)[0]
        sample[f"{model_alias}_pred"] = pred_text
        sample[f"{model_alias}_label"] = parse_prediction(pred_text, sample["input"])
        return sample

    return _batch_predict if batch else _predict
