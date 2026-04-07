import pandas as pd
import json
import re
import time
from openai import OpenAI
from string import Template

# import local
from src.schema import InjuryClassification
from src.prompt import ROLE, PROMPT
from src.utils import get_reference, get_narrative, get_kshot_examples

# Local vars
ROW_LIMIT = 500
FEATURE_NAME = "BDYPT"
REF_VARIABLE = "Body_Part"
TOP_LOGPROBS = 5
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# define AI model stuff
MODEL = "gpt-5.4-nano"
CLIENT = OpenAI()

# --------------------- #
# LOAD AND PROCESS DATA
# --------------------- #

neiss = pd.read_csv("data/neiss2024.csv", nrows=ROW_LIMIT)
neiss_ref = pd.read_csv("data/neiss2024_ref.csv")
neiss_examples = pd.read_csv("data/neiss2024_kshot_examples.csv")

# get refrence doc to map features back to tagged narratives
# and narratives with the tag of interest (e.g. body part)
# also pull some kshot examples
reference_doc = get_reference(neiss_ref, FEATURE_NAME)
tagged_narratives = get_narrative(neiss, REF_VARIABLE)
kshot_examples = get_kshot_examples(neiss_examples)

# --------------------- #
# SETUP PROMPT
# --------------------- #

# Create the template objects
role_template = Template(ROLE)
prompt_template = Template(PROMPT)

# populate role template with static examples
# this ensures we are also hitting input prompt caching
role = role_template.substitute(extraction_examples=kshot_examples)

# --------------------- #
# MODEL INVOCATION FUNCTIONS
# --------------------- #


def create_batch_request(idx, narrative):
    schema_str = json.dumps(InjuryClassification.model_json_schema(), indent=2)
    prompt = prompt_template.substitute(
        extraction_schema=schema_str,
        narrative=narrative,
    )

    return {
        "custom_id": f"req-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ],
        },
    }


def main():
    # --------------------- #
    # STEP 1: BUILD BATCH FILE
    # --------------------- #
    batch_file = f"artifacts/batch/batch_input_{TIMESTAMP}.jsonl"

    with open(batch_file, "w", encoding="utf-8") as f:
        for idx, single_narrative in enumerate(tagged_narratives):
            narrative_text = single_narrative["narrative"]

            # Construct the line using the invocation logic
            request_line = create_batch_request(idx, narrative_text)

            f.write(json.dumps(request_line) + "\n")

    print(f"Batch input file written: {batch_file}")

    # --------------------- #
    # STEP 2: SUBMIT BATCH
    # --------------------- #

    # Upload file
    with open(batch_file, "rb") as f:
        batch_input = CLIENT.files.create(file=f, purpose="batch")

    # Create job
    batch_job = CLIENT.batches.create(
        input_file_id=batch_input.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"Batch job submitted: {batch_job.id}")


# --------------------- #
# MAIN ENTRYPOINT
# --------------------- #

if __name__ == "__main__":
    main()
