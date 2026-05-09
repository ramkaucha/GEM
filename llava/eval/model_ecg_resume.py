import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import numpy as np
import time

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
import transformers
from typing import Dict
import re
from PIL import Image
import wfdb
import math


# Default folder layout assumed when --single-question is set.
# Folder must contain one .hea, .dat, and .png per record id (e.g. A00001.hea / A00001.dat / A00001.png).
DEFAULT_RECORDS_FOLDER = "./records"

# The single fixed question asked of every record, regardless of modality.
SINGLE_QUESTION_TEXT = (
    "Provide a clinical interpretation of this ECG. Include the rhythm, "
    "any conduction abnormalities, ST or T-wave findings, and an overall classification."
)


def build_questions_from_records(records_folder, modality):
    """Discover record ids in `records_folder` and build per-record question entries.

    Each record is expected to have:
      - <id>.hea / <id>.dat (signal, used by wfdb when modality in {ecg, both})
      - <id>.png            (image, used when modality in {image, both})

    For modality='both' we intersect the two sets so we only emit ids that have
    BOTH the signal and the image. For ecg-only or image-only modalities we use
    just the relevant file type.

    Returns a list of dicts shaped to match the existing question-file path so
    the rest of eval_model() doesn't need to special-case it.
    """
    if not os.path.isdir(records_folder):
        raise FileNotFoundError(f"Records folder not found: {records_folder}")

    files = os.listdir(records_folder)
    hea_ids = {os.path.splitext(fn)[0] for fn in files if fn.endswith(".hea")}
    png_ids = {os.path.splitext(fn)[0] for fn in files if fn.endswith(".png")}

    if modality == "ecg":
        record_ids = hea_ids
    elif modality == "image":
        record_ids = png_ids
    else:  # both
        record_ids = hea_ids & png_ids

    record_ids = sorted(record_ids)

    questions = []
    for rid in record_ids:
        # The existing eval loop expects:
        #   image  -> path fragment joined with args.image_folder
        #   ecg    -> path fragment joined with args.ecg_folder (wfdb uses base, no ext)
        # When --single-question is set we point both folders at records_folder, so
        # passing the bare record id for ecg and "<id>.png" for image works.
        questions.append({
            "question_id": rid,
            "image": f"{rid}.png",
            "ecg": rid,
            "text": SINGLE_QUESTION_TEXT,
            "ans": "",
            "conversations": [
                {"from": "human", "value": f"<image>\n{SINGLE_QUESTION_TEXT}"},
                {"from": "gpt", "value": ""},
            ],
        })
    return questions


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for sentence in source:
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum(i == IMAGE_TOKEN_INDEX for i in _input_id) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


def load_image_input(args, image_file, image_processor, model):
    if args.modality == "ecg":
        return None, None
    if not image_file:
        raise ValueError("Image input is required for image or both modality inference.")

    image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)[0]
    return image_tensor.unsqueeze(0).half().cuda(), [image.size]


def load_ecg_input(args, ecg_file):
    if args.modality == "image":
        return None
    if not ecg_file:
        raise ValueError("ECG input is required for ecg or both modality inference.")

    ecg = wfdb.rdsamp(os.path.join(args.ecg_folder, ecg_file))[0]
    ecg[np.isnan(ecg)] = 0
    ecg[np.isinf(ecg)] = 0
    ecg = torch.Tensor(np.transpose(ecg, (1, 0)).astype(np.float32))
    c, length = ecg.shape
    seq_length = 5000
    if length < seq_length:
        new_ecg = torch.zeros((c, seq_length))
        new_ecg[:, 0:length] = ecg
        ecg = new_ecg
    elif length > seq_length:
        ecg = ecg[:, 0:seq_length]
    return ecg.half()


def build_input_ids(model_name, tokenizer, line, prompt):
    if "qwen" in model_name.lower():
        conv_0 = dict(line["conversations"][0])
        if conv_0["value"] is None:
            conv_0["value"] = "<image>\n"
        elif "<image>" not in conv_0["value"]:
            conv_0["value"] = '<image>\n' + conv_0["value"]
        return preprocess_qwen([conv_0, {'from': 'gpt', 'value': None}], tokenizer, has_image=True).cuda()
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()


def eval_model(args):
    disable_torch_init()
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    if args.single_question:
        # In single-question mode, discover records on disk and use a fixed prompt.
        # Default both image-folder and ecg-folder to the records folder unless the
        # user explicitly overrode them on the CLI (so files line up with the names
        # we put in `image` / `ecg` inside build_questions_from_records).
        if not args.image_folder:
            args.image_folder = args.records_folder
        if not args.ecg_folder:
            args.ecg_folder = args.records_folder
        questions = build_questions_from_records(args.records_folder, args.modality)
        print(f"[single-question] Found {len(questions)} records under {args.records_folder} "
              f"for modality={args.modality}")
    else:
        questions = []
        with open(args.question_file, "r") as f:
            json_data = json.load(f)
            for line in json_data:
                questions.append({
                    "question_id": line["id"],
                    "image": line["image"],
                    "text": line["conversations"][0]["value"].replace("<image>\n", ""),
                    "ans": line["conversations"][1]["value"],
                    "ecg": line["ecg"],
                    "conversations": line["conversations"],
                })

    # Dedup by (question_id, modality) so multiple runs over the same JSONL
    # (one per modality) don't skip each other.
    existing_keys = set()
    if os.path.exists(args.answers_file):
        with open(args.answers_file, "r") as ans_file:
            for line in ans_file:
                line = line.strip()
                if not line:
                    continue
                existing_data = json.loads(line)
                prior_modality = existing_data.get("metadata", {}).get("modality", "")
                existing_keys.add((existing_data["question_id"], prior_modality))

    # Make sure the answers directory exists (the bash wrapper expects this).
    answers_dir = os.path.dirname(args.answers_file)
    if answers_dir:
        os.makedirs(answers_dir, exist_ok=True)
    output_file = open(args.answers_file, "a")

    for line in tqdm(questions):
        idx = line["question_id"]
        if (idx, args.modality) in existing_keys:
            print(f"Skipping question {idx} for modality {args.modality}, already exists.")
            continue

        image_file = line["image"]
        ecg_file = line["ecg"]
        qs = line["text"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = build_input_ids(model_name, tokenizer, line, prompt)
        images, image_sizes = load_image_input(args, image_file, image_processor, model)
        ecg = load_ecg_input(args, ecg_file)

        start_gen = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                ecgs=ecg,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        torch.cuda.synchronize()
        end_gen = time.perf_counter()

        print(f"Generation_latency_s={end_gen - start_gen:.4f}")

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        new_answer = {
            "question_id": idx,
            "record_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {"modality": args.modality},
        }

        output_file.write(json.dumps(new_answer) + "\n")
        output_file.flush()

    output_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--ecg-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--ecg_tower", type=str, default="")
    parser.add_argument("--open_clip_config", type=str, default="")
    parser.add_argument("--modality", type=str, default="both", choices=["both", "ecg", "image"])

    # Single-question mode: skip --question-file, ask one fixed clinical-interpretation
    # question for every record found under --records-folder. Output rows still carry
    # the modality in metadata.modality so a downstream eval script can group by it.
    parser.add_argument(
        "--single-question",
        action="store_true",
        help="Discover records under --records-folder and ask SINGLE_QUESTION_TEXT for each.",
    )
    parser.add_argument(
        "--records-folder",
        type=str,
        default=DEFAULT_RECORDS_FOLDER,
        help="Folder containing <id>.hea/.dat/.png triples. Used when --single-question is set.",
    )

    args = parser.parse_args()
    eval_model(args)
