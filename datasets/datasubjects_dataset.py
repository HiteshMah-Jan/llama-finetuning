import datasets

from llama_recipes.datasets.utils import Concatenator


B_INST, E_INST = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_CODE, E_CODE = "\n[CODE]\n", "\n[/CODE]\n"
B_FILE, E_FILE = "\n[FILE]\n", "\n[/FILE]\n"
B_DTSB, E_DTSB = "\n[DTSB]\n", "\n[/DTSB]\n"
NEW_LINE = "\n"


def format_text(row, tokenizer):
    text = (
        B_INST
        + B_SYS
        + "You are an Privacy Engineer having a lot of experience developing large codebases.\n"
        + "A `data subject` refers to an identifiable individual as defined by GDPR and related privacy laws. An individual can be identified directly or indirectly using various identifiers, be they personal data elements, PIIs (Personally Identifiable Information), or PHIs (Personal Health Information).\n"
        + E_SYS
        + B_CODE
        + row["code"]
        + "Filename : " + row["fileName"]
        + "Match : " + row["match"]
        + "Data Element : " + row["name"]
        + E_CODE
        + "Analyze the given `code` snippet, the highlighted `match`, and the `filename`. Extract potential data subject or individual/group that the `match` could represent. If there are multiple potential matches, return a single most relevant Data Subject.\n"
        + E_INST
        + B_DTSB
        + row["dataSubject"]
        + E_DTSB
        + "</s>"
    )

    return tokenizer(text)


def get_custom_dataset(dataset_config, tokenizer, split):
    full_dataset = datasets.load_dataset("Privado/data-subjects", split="train")

    # Since the dataset has no train/test split, we create one and select it
    dataset = full_dataset.train_test_split(
        train_size=0.95,
        test_size=0.05,
        seed=42,
    )["train" if split == dataset_config.train_split else "test"]

    dataset = dataset.map(
        lambda x: format_text(x, tokenizer), remove_columns=list(dataset.features)
    )

    dataset = dataset.map(Concatenator(), batched=True, batch_size=None)

    return dataset
