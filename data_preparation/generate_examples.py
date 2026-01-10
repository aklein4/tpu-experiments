
import datasets
from tqdm import tqdm
import os

URL = "aklein4/compilation-SmolLM2"
OUTPUT_DIR = "data_statistics"

NUM_EXAMPLES = 10


def main():
    
    subsets = list(datasets.get_dataset_config_names(URL))

    for subset in tqdm(subsets):

        with open(os.path.join(OUTPUT_DIR, subset, "examples.txt"), "w", encoding="utf-8") as f:

            dataset = datasets.load_dataset(URL, subset, split="train", streaming=True)

            for i, example in enumerate(dataset):
                f.write(f"\n\n ========== Example {i} ==========")

                f.write("\n\n --- Input --- \n\n")
                f.write("[[["+example["input"]+"]]]")

                f.write("\n\n --- Output --- \n\n")
                f.write("[[["+example["output"]+"]]]")

                if i + 1 >= NUM_EXAMPLES:
                    break


if __name__ == "__main__":
    main()