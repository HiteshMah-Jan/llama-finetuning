import datasets

from llama_recipes.datasets.utils import Concatenator

B_INST, E_INST = "[INST] ", " [/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_CODE, E_CODE = "\n[CODE]\n", "\n[/CODE]\n"
B_DTSB, E_DTSB = "\n[DTSB]\n", "\n[/DTSB]\n"
NEW_LINE = "\n"

EXAMPLE_DATA_SUBJECTS = [
    '- Doctor (synonyms: Physician, Specialist, Surgeon)',
    '- Patient (synonyms: Care Recipient)',
    '- Secretary (synonyms: Administrative Assistant, Clerk)',
    '- Receptionist (synonyms: Front Desk, Welcome Staff)',
    '- User (synonyms: End User, Consumer)',
    '- Policyholder',
    '- Claimant',
    '- Employee (synonyms: Worker, Staff)',
    '- Guardian (synonyms: Custodian, Protector, Parent)',
    '- Student (synonyms: Learner, Pupil)',
    '- Partner (synonyms: Collaborator, Associate)',
    '- Seller (synonyms: Vendor, Merchant)',
    '- Buyer (synonyms: Purchaser, Client)'
]

def format_text(row, tokenizer):
    text = (
        B_INST
        + B_SYS
        + "A `data subject` refers to an identifiable individual as defined by GDPR and related privacy laws. An individual can be identified directly or indirectly using various identifiers, be they personal data elements, PIIs (Personally Identifiable Information), or PHIs (Personal Health Information).\n"
        + "Examples of data subjects include, but are not limited to: "
        + NEW_LINE + NEW_LINE.join(EXAMPLE_DATA_SUBJECTS) + NEW_LINE
        + "Note : The above list of data subjects and their corresponding synonyms are not exhaustive, and there can be other data subjects and synonyms not mentioned above."
        + E_SYS
        + B_CODE
        + "Repository : " + row["repoName"]
        + "DataElement : " + row["name"]
        + "Match : " + row["match"]
        + "Filename : " + row["fileName"]
        + "Code : ```" + row["code"] + "```"
        + E_CODE
        + "Given the `Code` snippet found in a specific `Filename`, identify the potential data subject or individual that the highlighted `Match` found in `Code` could represent, especially in the context of the associated `DataElement` which represents a specific personal data element, PII, or PHI. Understand the interplay between `Code`, its `Filename`, and the nature of the `Match` to determine the most probable data subject. If a synonym is identified, associate it with a primary label. If no specific data subject can be identified, return the label 'NA'. Return a single most relevant Data Subject.\n"
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
