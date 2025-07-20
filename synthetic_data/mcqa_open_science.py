

import datasets

INPUT_REPO = "nvidia/OpenScience"
SUBSETS = ["OS-Q2.5-32B-4", "OS-Q3-235B-4"]


def format_question(question):
    
    if '?' not in question:
        raise ValueError("Question must contain a '?' character.")

    q, answers = question.split('?')
    q = q.strip() + '?'

    answers = [answer.strip() for answer in answers.strip().split('\n')]
    try:
        for i, a in enumerate(answers):
            answers[i] = a[0] + '.' + a[2:]
    except:
        raise ValueError("Answers must be formatted as 'A. answer'.")

    out = f"Question:\n{q}\nChoices:"
    for a in answers:
        out += f"\n{a}"
    
    return out


def format_answer(output):
    output = output.split("</think>")[1].strip()

    if len(output) < 50:
        raise ValueError("Output is too short to be a valid answer.")

    correct_ending = False
    for letter in ['A', 'B', 'C', 'D']:
        if output.endswith("\\boxed{"+letter+"}"):
            correct_ending = True
            break
    if not correct_ending:
        raise ValueError("Output does not end with a valid answer format.")
    
    answer = output[-2]

    return output, answer


def map_example(example):

    try:
        question = format_question(example["input"])
        output, answer = format_answer(example["output"])

        return {
            "source": INPUT_REPO,
            "question": question,
            "answer": answer,
            "explanation": output,
        }

    except ValueError:
        return {
            "source": INPUT_REPO,
            "question": None,
            "answer": None,
            "explanation": None,
        }


def main():
    
    subs = []
    for subset in SUBSETS:
        ds = datasets.load_dataset(INPUT_REPO, subset, split="train")
        subs.append(ds)
    dataset = datasets.concatenate_datasets(subs)

    dataset = dataset.map(
        map_example,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.filter(lambda x: x["question"] is not None)

    print(f"Loaded {len(dataset)} examples from {INPUT_REPO}!")

    dataset.push_to_hub(
        "aklein4/open-science-formatted",
        split="train",
    )


if __name__ == "__main__":
    
    # mcqa = datasets.load_dataset('aklein4/mcqa-synthetic-explanations-lite', split='train')
    # mcqa = mcqa.map(
    #     lambda x: {"source": x["source"]+"-synthetic-explanations", "explanation": x["explaination"]},
    #     remove_columns=["explaination"]
    # )

    # os = datasets.load_dataset('aklein4/open-science-formatted', split='train')

    # combined = datasets.concatenate_datasets([mcqa, os])
    # combined = combined.shuffle(seed=42)

    # combined.push_to_hub(
    #     "mcqa",
    #     split="train",
    # )

    main()
    