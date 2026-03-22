# interactive_gem_cli.py
import argparse
import os
import time
from threading import Thread

import numpy as np
import torch
import wfdb
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_ecg_tensor(record_path: str) -> torch.Tensor:
    ecg = wfdb.rdsamp(record_path)[0]
    ecg[np.isnan(ecg)] = 0
    ecg[np.isinf(ecg)] = 0
    ecg = torch.tensor(np.transpose(ecg, (1, 0)).astype(np.float32))

    seq_length = 5000
    c, length = ecg.shape
    if length < seq_length:
        new_ecg = torch.zeros((c, seq_length), dtype=torch.float32)
        new_ecg[:, :length] = ecg
        ecg = new_ecg
    elif length > seq_length:
        ecg = ecg[:, :seq_length]

    return ecg.half()


def build_first_user_message(model, user_text: str) -> str:
    if model.config.mm_use_im_start_end:
        return DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_text
    return DEFAULT_IMAGE_TOKEN + "\n" + user_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--ecg-record", required=True, help="WFDB record stem path, without .hea")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    disable_torch_init()

    print("Loading model...")
    load_t0 = time.perf_counter()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    load_t1 = time.perf_counter()

    print("Loading image and ECG...")
    prep_t0 = time.perf_counter()
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0).half().cuda()

    ecg = load_ecg_tensor(args.ecg_record).cuda()
    prep_t1 = time.perf_counter()

    conv = conv_templates[args.conv_mode].copy()

    print(f"Model load time: {load_t1 - load_t0:.2f}s")
    print(f"Input prep time: {prep_t1 - prep_t0:.2f}s")
    print("Interactive session started.")
    print("Commands: /reset, /timing, /exit")
    print()

    total_turns = 0

    while True:
        user_text = input("USER> ").strip()
        if not user_text:
            continue
        if user_text == "/exit":
            break
        if user_text == "/reset":
            conv = conv_templates[args.conv_mode].copy()
            print("Conversation reset.")
            continue
        if user_text == "/timing":
            print("Timing note: startup cost is one-time; only per-turn generation matters after load.")
            continue

        if len(conv.messages) == 0:
            prompt_text = build_first_user_message(model, user_text)
        else:
            prompt_text = user_text

        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            inputs=input_ids,
            ecgs=ecg,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=(args.temperature > 0),
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            streamer=streamer,
        )

        turn_t0 = time.perf_counter()
        first_token_time = None

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        print("ASSISTANT> ", end="", flush=True)
        pieces = []
        for new_text in streamer:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if stop_str and new_text.endswith(stop_str):
                new_text = new_text[: -len(stop_str)]
            print(new_text, end="", flush=True)
            pieces.append(new_text)

        thread.join()
        turn_t1 = time.perf_counter()
        print()

        answer = "".join(pieces).strip()
        conv.messages[-1][1] = answer

        total_turns += 1
        ttft = None if first_token_time is None else first_token_time - turn_t0
        total_latency = turn_t1 - turn_t0
        print(
            f"[turn {total_turns}] total={total_latency:.2f}s"
            + (f", first_token={ttft:.2f}s" if ttft is not None else "")
        )
        print()


if __name__ == "__main__":
    main()
