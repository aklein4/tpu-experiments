
import random

from handlers import HANDLERS


LOG_FILE = "compilation_log.txt"

MAX_INPUT_CHARACTERS = 1000 * 10
MAX_OUTPUT_CHARACTERS = 1000 * 10

NAMES_TO_DO = None


def main():
    
    random.seed(42)

    with open(LOG_FILE, "w") as f:
        f.write("")

    handler_list = HANDLERS
    if NAMES_TO_DO is not None:
        handler_list = [
            h for h in handler_list if h().name() in NAMES_TO_DO
        ]

    total_examples = 0
    for i, h_type in enumerate(handler_list):

        h = h_type(
            max_input_characters=MAX_INPUT_CHARACTERS,
            max_output_characters=MAX_OUTPUT_CHARACTERS,
        )

        print("")
        print(f"[{i+1}/{len(HANDLERS)}] Processing dataset: {h.name()}")
        print("")

        try:

            ds = h.load_dataset()
            
            ds = ds.map(h.full_map, remove_columns=ds.column_names, load_from_cache_file=False)
            ds = ds.filter(h.filter, load_from_cache_file=False)
        
            ds.push_to_hub(
                "aklein4/raw-compilation",
                config_name=h.name().replace("/", "--"),
                private=False,
                split="train",
            )

        except:
            with open(LOG_FILE, "a") as f:
                f.write(f"\n{h.name()}: FAIL")
            continue

        with open(LOG_FILE, "a") as f:
            f.write(f"\n{h.name()}: SUCCESS ({len(ds)} examples)")
        total_examples += len(ds)
    
    with open(LOG_FILE, "a") as f:
        f.write(f"\n\nTotal examples: {total_examples}\n")


if __name__ == "__main__":
    main()