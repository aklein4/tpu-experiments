
import datasets
import random
from functools import partial
import json


def combine_system_prompt(data):
    if data[0]["role"] == "system":
        first = {
            "content": data[0]["content"].strip() + '\n' + data[1]["content"].strip(),
            "role": "user"
        }

        data = [first] + data[2:]

    return data


def compilation_map(example):
    data = example["data"]

    data = combine_system_prompt(data)
    
    inp = f"Question:\n{data[0]["content"].strip()}\nAnswer:\n"
    out = data[1]["content"].strip()

    return {
        "source": example["source"],
        "input": inp,
        "output": out,
    }


def synth_map(example, subset=""):
    data = example["message"]

    data = combine_system_prompt(data)

    inp = f"Question:\n{data[0]["content"].strip()}\nAnswer:\n"
    out = data[1]["content"].strip()

    return {
        "source": f'IgnoraZ/SynthQuestions/{subset}',
        "input": inp,
        "output": out,
    }


def webinstruct_map(example):
    return {
        "source": "TIGER-Lab/WebInstructSub",
        "input": f"Question:\n{example["question"].strip()}\nAnswer:\n",
        "output": example["answer"].strip(),
    }


def main():
    
    compilation = datasets.load_dataset("aklein4/chat-compilation", split="train")

    with open("C:/Users/adam3/Downloads/realquestions.jsonl", "r", encoding="utf-8") as f:
        l_real = [json.loads(line) for line in f.readlines()]
    realquestions = datasets.Dataset.from_list(l_real)
    
    with open("C:/Users/adam3/Downloads/synthquestions_1m.moderated.jsonl", "r", encoding="utf-8") as f:
        l_synth = [json.loads(line) for line in f.readlines()]
    synthquestions = datasets.Dataset.from_list(l_synth)

    webinstruct = datasets.load_dataset("TIGER-Lab/WebInstructSub", split="train")

    compilation = compilation.filter(
        lambda x: x["source"] not in ["facebook/natural_reasoning", "lmsys/lmsys-chat-1m"]
    )

    compilation = compilation.map(
        compilation_map,
        remove_columns=compilation.column_names,
    )
    realquestions = realquestions.map(
        synth_map,
        remove_columns=realquestions.column_names,
        fn_kwargs={"subset": "realquestions"},
    )
    synthquestions = synthquestions.map(
        synth_map,
        remove_columns=synthquestions.column_names,
        fn_kwargs={"subset": "synthquestions"},
    )
    webinstruct = webinstruct.map(
        webinstruct_map,
        remove_columns=webinstruct.column_names,
    )

    combined = datasets.concatenate_datasets(
        [
            compilation,
            realquestions,
            synthquestions,
            webinstruct
        ]
    )
    shuffled = combined.shuffle(seed=42)

    shuffled.push_to_hub(
        "chat-formatted",
    )


if __name__ == "__main__":
    main()