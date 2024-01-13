import argparse
import json
import logging
import pathlib
import subprocess
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATA_ROOT = (pathlib.Path(__file__).parent / "data").resolve()


def main(config):
    # Create QA data
    with open('config.json', 'r') as config_file:
        config_data = json.load(config_file)

    dataset_paths = config_data['datasets']

    logger.info(f"Creating QA data for datasets {[d for d in dataset_paths]}...")
    for dataset in tqdm(dataset_paths):
        for i, n_hop in enumerate(config_data['datasets'][dataset]["n_hops"]):
            if n_hop == "closedbook":
                input_args = [
                    "-i", DATA_ROOT / ("base/" + config_data['datasets'][dataset]["file_name"]),
                    "-d", dataset,
                    "-p", "qa_closedbook.prompt"]
                subprocess.run([sys.executable, 'scripts/make_qa_prompts.py'] + input_args)
            else:
                for pos in config_data['datasets'][dataset]["positions"][i]:
                    input_args = [
                        "-i", DATA_ROOT / ("base/" + config_data['datasets'][dataset]["file_name"]),
                        "--n_hops", str(n_hop),
                        "-g", ",".join([str(p) for p in pos]),
                        "-d", dataset,
                        "-p", config_data['datasets'][dataset]["prompt_file"],
                        "--doctype", "Document"
                    ]

                    subprocess.run([sys.executable, 'scripts/make_qa_prompts.py'] + input_args)

    # Get responses from model
    input_args = ["-i"]
    for file_path in pathlib.Path(DATA_ROOT / "generated").iterdir():
        if file_path.is_file():
            input_args.append(str(file_path))

    subprocess.run([sys.executable, 'scripts/get_mpt_responses.py'] + input_args)




if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", "-c",
        help=("Path to experiment configuration."),
        required=True,
    )


    args = parser.parse_args()


    logger.info("running %s", " ".join(sys.argv))
    main(args.config)
    logger.info("finished running %s", sys.argv[0])


