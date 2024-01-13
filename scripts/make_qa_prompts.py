#!/usr/bin/env python3
"""
Running:

```
python -u ./scripts/make_qa_data_prompts.py \

    --gold-indices 0,1 \

```

"""
import argparse
import json
import logging
import sys
from copy import deepcopy
import pathlib
from xopen import xopen

logger = logging.getLogger(__name__)

PROMPTS_ROOT = (pathlib.Path(__file__).parent.parent / "prompts").resolve()
DATA_ROOT = (pathlib.Path(__file__).parent.parent / "data").resolve()


def main(input_path, gold_indices, output_path, doctype, prompt_filename):

    num_output_examples = 0

    with xopen(input_path) as fin, xopen(output_path, "w") as fout:

        data = json.load(fin)

        with open(PROMPTS_ROOT / prompt_filename) as f:
            prompt_template = f.read().rstrip("\n")

        if prompt_filename == "qa_closedbook.prompt":
            for i in range(len(data['question'])):
                prompt = prompt_template.format(question=data['question'][str(i)])
                fout.write(json.dumps(prompt) + "\n")
                num_output_examples += 1
        else:
            for i in range(len(data['question'])):
                distractors = [(data['distractor_docs'][str(i)]['title'][j], data['distractor_docs'][str(i)]['paragraph'][j]) for j in range(len(data['distractor_docs'][str(i)]['title']))]
                golds = [(data['gold_docs'][str(i)]['title'][j], data['gold_docs'][str(i)]['paragraph'][j]) for j in range(len(data['gold_docs'][str(i)]['title']))]
                if len(golds) != len(gold_indices):
                    continue

                ordered_docs = order_input(distractors, golds, gold_indices)

                formatted_docs = "\n".join([f"{doctype} [{j + 1}](Title: {ordered_docs[j][0]}) {ordered_docs[j][1]}" for j in range(len(ordered_docs))])

                prompt = prompt_template.format(question=data['question'][str(i)], search_results=formatted_docs)

                fout.write(json.dumps(prompt) + "\n")
                num_output_examples += 1
    #logger.info(f"Wrote {num_output_examples} output examples")


def order_input(distractors, golds, gold_positions):
    distractors = deepcopy(distractors)
    context = []
    total_len = len(distractors) + len(golds)
    golds_len = len(golds)
    i = 0
    while i < total_len:
        if i in gold_positions:
            context.append(golds[gold_positions.index(i)])
            golds_len -= 1
        else:
            if len(distractors) > 0:
                context.append(distractors.pop(0))

        i += 1

    # For rare cases where hotpot had fewer than 20 docs:
    # place any remaining gold evidence at the end
    while golds_len > 0:
        context.append(golds[2 - golds_len])
        golds_len -= 1

    return context


if __name__ == "__main__":
    #logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_path", "-i",
        help=("Path to base dataset."),
        required=True,
    )

    parser.add_argument(
        '--n_hops',
        help='Which split to use based on number of hops.',
        type=int,
        )

    parser.add_argument(
        '--gold_indices', '-g',
        help='Indices to place gold documents at, separated by commas.',
        type=str,
        )

    parser.add_argument(
        '--dataset', '-d',
        help='Dataset to use. Choices include \'hotpot\', \'2wiki\', and \'musique\'',
        type=str,
        choices=['hotpot', '2wiki', 'musique'],
        required=True)

    parser.add_argument(
        '--prompt', '-p',
        help='Prompt file name.',
        type=str,
        required=True)

    parser.add_argument(
        '--doctype',
        help='Optional document type to be used in prompts. Defaults to \'Document\'.',
        type=str,
        default='Document'
        )
    parser.add_argument(
        '--uses_doctype',
        help='Whether to include document type in the prompts. Defaults to True.',
        type=bool,
        default=True,
    )

    parser.add_argument("--output", '-o', help="Path to write output data files (optional).")

    args = parser.parse_args()

    if args.prompt != "qa_closedbook.prompt":
        args.gold_indices = [int(i) for i in args.gold_indices.split(',')]

        assert args.gold_indices is not None, "Selected prompt requires --gold_indices."
        assert args.n_hops is not None, "Selected prompt requires --n_hops."
        assert int(args.n_hops) == len(args.gold_indices), "--n_hops should be equal to the number of --gold_indices"
        if args.dataset == 'hotpot':
            assert args.n_hops == 2, "Dataset doesn't contain specified number of hops. Valid hops include [2]."
        elif args.dataset == '2wiki':
            assert args.n_hops in [2, 4], "Dataset doesn't contain specified number of hops. Valid hops include [2, 4]."
        elif args.dataset == 'musique':
            assert args.n_hops in [2, 3,
                                   4], "Dataset doesn't contain specified number of hops. Valid hops include [2, 3, 4]."


        if args.output is None:
            args.output = DATA_ROOT / f"generated/{args.dataset}_{args.n_hops}hop_{'_'.join([str(i) for i in args.gold_indices])}.json"
    else:
        if args.output is None:
            args.output = DATA_ROOT / f"generated/{args.dataset}_closedbook.json"

    if pathlib.Path(args.output).exists():
        logger.warning(f"File {args.output} already exists. Skipping.")
        sys.exit(1)





    #logger.info("running %s", " ".join(sys.argv))
    main(args.input_path, args.gold_indices, args.output, args.doctype if args.uses_doctype else None, args.prompt)
    #logger.info("finished running %s", sys.argv[0])