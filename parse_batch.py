import pandas as pd
import json

from src.schema import InjuryClassification
from src.utils import get_narrative, get_kshot_examples, resolve_result

ROW_LIMIT = 500
FEATURE_NAME = "BDYPT"
REF_VARIABLE = "Body_Part"

def _token_matches_value(token_text: str, value: str) -> bool:
    return token_text == value or token_text.strip().strip('"').strip(",") == value

def extract_body_part_logprobs(choice, predicted_body_part: str):
    content_logprobs = getattr(choice, "logprobs", {}).get("content", []) if isinstance(choice, dict) else []

    token_record = None

    for i, token_info in enumerate(content_logprobs):
        if "body_part" in token_info.get("token", ""):
            for next_token in content_logprobs[i + 1 : i + 8]:
                if _token_matches_value(next_token.get("token", ""), predicted_body_part):
                    token_record = next_token
                    break
        if token_record:
            break

    if token_record is None:
        for token_info in content_logprobs:
            if _token_matches_value(token_info.get("token", ""), predicted_body_part):
                token_record = token_info
                break

    if token_record is None:
        return None

    return {
        "token": token_record.get("token"),
        "logprob": token_record.get("logprob"),
        "top_logprobs": token_record.get("top_logprobs"),
    }

neiss = pd.read_csv("data/neiss2024.csv", nrows=ROW_LIMIT)
neiss_ref = pd.read_csv("data/neiss2024_ref.csv")
neiss_examples = pd.read_csv("data/neiss2024_kshot_examples.csv")

tagged_narratives = get_narrative(neiss, REF_VARIABLE)
kshot_examples = get_kshot_examples(neiss_examples)

output_path = f"/home/gmcirco/Documents/Projects/injury-classifier/artifacts/batch/batch_69d3b2549b4081908fdd08ad28001c50_output.jsonl"

def extract_body_part_logprobs(choice, predicted_body_part: str):
    # Ensure we are accessing the logprobs correctly from the choice dictionary
    logprobs_node = choice.get("logprobs", {})
    if not logprobs_node:
        return {}

    content_logprobs = logprobs_node.get("content", [])
    token_record = None

    # Step 1: Find the token record for the predicted body part
    for i, token_info in enumerate(content_logprobs):
        if "body_part" in token_info.get("token", ""):
            # Look ahead a few tokens to find the actual value (e.g., "30")
            for next_token in content_logprobs[i + 1 : i + 8]:
                if _token_matches_value(next_token.get("token", ""), predicted_body_part):
                    token_record = next_token
                    break
        if token_record:
            break

    # Fallback: Just look for the first instance of the predicted value
    if token_record is None:
        for token_info in content_logprobs:
            if _token_matches_value(token_info.get("token", ""), predicted_body_part):
                token_record = token_info
                break

    if token_record is None:
        return {}

    # Step 2: Flatten the top_logprobs into a dictionary
    flattened = {}
    top_entries = token_record.get("top_logprobs", [])
    
    for i, entry in enumerate(top_entries, 1):
        label = entry.get("token", "").strip().replace('"', '')
        lp_val = entry.get("logprob")
        
        flattened[f"body_part_logprob_label_{i}"] = label
        flattened[f"body_part_logprob_prob_{i}"] = lp_val
        # Optional: flattened[f"body_part_linear_prob_{i}"] = math.exp(lp_val)

    return flattened

res_list = []

with open(output_path, "r") as f:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)

        idx = int(obj["custom_id"].split("-")[1])
        original = tagged_narratives[idx]

        try:
            choice = obj["response"]["body"]["choices"][0]
            content = choice["message"]["content"]
            parsed = InjuryClassification.model_validate_json(content)

            # Get the flattened logprob columns
            bp_logprob_cols = extract_body_part_logprobs(
                choice, parsed.body_part.value
            )

            is_correct = resolve_result(
                original["ref"], parsed.body_part.value
            )

            # Build the row dictionary
            row = {
                "original_ref": original["ref"],
                "predicted_part": parsed.body_part.value,
                "is_correct": is_correct,
                "reasoning": parsed.reasoning,
                "confidence": parsed.confidence,
            }
            
            # Merge the logprob columns into the row
            row.update(bp_logprob_cols)
            res_list.append(row)

        except Exception as e:
            print(f"Parse error on custom_id {obj.get('custom_id')}: {e}")

accuracy = sum(r["is_correct"] for r in res_list) / len(res_list)
print(f"Accuracy: {accuracy:.2%}")

import time

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
df = pd.DataFrame(res_list)
csv_path = f"artifacts/output/narrative_results_{TIMESTAMP}.csv"
df.to_csv(csv_path, index=False)