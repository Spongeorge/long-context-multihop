import argparse
import dataclasses
import json
import logging
import math
import pathlib
import random
import sys
from copy import deepcopy

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from xopen import xopen

logger = logging.getLogger(__name__)

OUTPUT_ROOT = (pathlib.Path(__file__).parent.parent / "output").resolve()


def main(
    input_paths,
    model_name,
    temperature,
    batch_size,
    num_gpus,
    max_memory_per_gpu,
    max_new_tokens,
    output_dir,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.attn_config["attn_impl"] = "triton"
    config.max_seq_len = 8192

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model")
    if num_gpus > 1:
        extra_kwargs = {
            "device_map": "auto",
            "max_memory": {i: f"{str(max_memory_per_gpu)}GiB" for i in range(num_gpus)},
        }
    else:
        extra_kwargs = {}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
        low_cpu_mem_usage=True,
        load_in_8bit=True,
    )

    #if num_gpus == 1:
    #    model = model.to(device)

    print(model)

    # Iterate over input files and generate responses
    for path in input_paths:
        logger.info(f"Running {path}")
        output_path = output_dir / f"{model_name.split('/')[-1]}_{pathlib.Path(path).name}"

        if pathlib.Path(output_path).exists():
            logger.warning(f"File {output_path} already exists. Skipping.")
            continue

        with xopen(path) as fin:
            prompts = fin.readlines()

        responses = []
        with torch.autocast(device, dtype=torch.bfloat16):
            for batched_prompts in tqdm(chunks(prompts, batch_size), total=math.ceil(len(prompts) / batch_size)):
                inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    use_cache=True,
                    eos_token_id=0,
                    pad_token_id=0,
                    return_dict_in_generate=False,
                )
                for i, generated_sequence in enumerate(outputs):
                    input_ids = inputs[i].ids
                    text = tokenizer.decode(generated_sequence, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=True)

                    if input_ids is None:
                        prompt_length = 0
                    else:
                        prompt_length = len(
                            tokenizer.decode(
                                input_ids,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True,
                            )
                        )
                    new_text = text[prompt_length:]
                    responses.append(new_text)

        with open(output_path, "w") as f:
            f.write(json.dumps(responses))




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", "-i",
        help=("Path to QA data. Multiple files allowed."),
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--model_name", "-m",
        default="mosaicml/mpt-7b-8k-instruct",
    )
    parser.add_argument(
        "--temperature", "-t",
        default=0,
    )
    parser.add_argument(
        "--max_new_tokens",
        default=256,
    )
    parser.add_argument("--batch-size", help="Batch size use in generation", type=int, default=4)

    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument(
        "--max-memory-per-gpu",
        help="Maximum memory to use per GPU (in GiB) for multi-device parallelism, e.g., 80",
        type=int,
    )

    parser.add_argument("--output_dir", '-o', help="Path to write output data files (optional).")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = OUTPUT_ROOT / ""


    #logger.info("running %s", " ".join(sys.argv))
    main(args.input_path,
         args.model_name,
         args.temperature,
         args.batch_size,
         args.num_gpus,
         args.max_memory_per_gpu,
         args.max_new_tokens,
         args.output_dir
         )
    #logger.info("finished running %s", sys.argv[0])


