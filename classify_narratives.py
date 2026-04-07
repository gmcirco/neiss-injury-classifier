import pandas as pd
import json
import re
import time
from openai import OpenAI
from string import Template
from pydantic import ValidationError

# import local
from src.schema import InjuryClassification
from src.prompt import ROLE, PROMPT
from src.utils import (
    get_reference,
    get_narrative,
    get_kshot_examples,
    extract_body_part_logprobs,
    resolve_result,
)

# Local vars
ROW_LIMIT = 50
FEATURE_NAME = "BDYPT"
REF_VARIABLE = "Body_Part"
TOP_LOGPROBS = 5
TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

# define AI model stuff
MODEL = "gpt-5.4-nano"
CLIENT = client = OpenAI()

# --------------------- #
# LOAD AND PROCESS DATA
# --------------------- #

neiss = pd.read_csv("data/neiss2024.csv", nrows=ROW_LIMIT, skiprows=range(1,3000))
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

def invoke(client, narrative):
    schema_str = json.dumps(InjuryClassification.model_json_schema(), indent=2)
    prompt = prompt_template.substitute(
        extraction_schema=schema_str,
        narrative=narrative,
    )
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt},
            ],
            model=MODEL,
            temperature=0,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content
        parsed = InjuryClassification.model_validate_json(raw_content)
        bp_logprobs = extract_body_part_logprobs(
            response.choices[0], parsed.body_part.value
        )
        return parsed, bp_logprobs
    except (ValidationError, Exception) as e:
        print(f"Error during LLM invocation or validation: {e}")
        return None


# Parse a single complaint
def process_narrative(narrative_text):
    invoke_result = invoke(CLIENT, narrative_text)

    if invoke_result is None:
        return None

    response, bp_logprobs = invoke_result

    return {
        "body_part": response.body_part.value,
        "reasoning": response.reasoning,
        "confidence": response.confidence,
        "body_part_logprobs": bp_logprobs,
    }


def main():
    res_list = []

    for single_narrative in tagged_narratives:
        narrative_text = single_narrative["narrative"]
        body_part_tag = single_narrative["ref"]

        # Process the narrative
        res = process_narrative(narrative_text)

        # Check if the classification is correct
        is_correct = resolve_result(body_part_tag, res["body_part"])

        # for debugging, print results in terminal
        print(
            f'PREDICTED: {res["body_part"]}\nCONFIDENCE: {res['confidence']}\nREASONING: {res['reasoning']}\nCORRECT: {is_correct}\n'
        )

        # Append a structured dictionary
        res_list.append(
            {
                "narrative": narrative_text,
                "original_ref": body_part_tag,
                "predicted_part": res["body_part"],
                "is_correct": is_correct,
                "reasoning": res["reasoning"],
                "confidence": res["confidence"],
                "body_part_logprobs": res["body_part_logprobs"],
            }
        )

    # Quick Summary Example:
    accuracy = sum(item["is_correct"] for item in res_list) / len(res_list)
    print(f"Accuracy: {accuracy:.2%}")

    # EXPORT
    output_file = f"artifacts/output/narrative_results_{TIMESTAMP}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(res_list, f, indent=4)

# --------------------- #
# MAIN ENTRYPOINT
# --------------------- #

if __name__ == "__main__":
    main()