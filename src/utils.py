import re

# local functions
def get_reference(file, feature_name):
    return file[file["Format name"] == feature_name]


def get_narrative(file, reference_variable):
    return [
        {"narrative": n, "ref": t}
        for n, t in zip(file["Narrative_1"], file[reference_variable])
    ]


def get_kshot_examples(file):
    formatted_list = [
        f"\n\n[BODY_PART]: {row.Body_Part}\n[NARRATIVE]: {row.Narrative_1.strip()}"
        for row in file.itertuples()
    ]
    return "".join(formatted_list)


def resolve_result(observed, predicted):
    if match := re.search(r"\b\d{1,2}\b", str(predicted)):
        return str(observed) == match.group()
    return False


def _token_matches_value(token_text: str, value: str) -> bool:
    return token_text == value or token_text.strip().strip('"').strip(",") == value


def extract_body_part_logprobs(choice, predicted_body_part: str):
    content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None) or []
    token_record = None

    # Prefer the token that appears right after "body_part" in JSON.
    for i, token_info in enumerate(content_logprobs):
        if "body_part" in token_info.token:
            for next_token in content_logprobs[i + 1 : i + 8]:
                if _token_matches_value(next_token.token, predicted_body_part):
                    token_record = next_token
                    break
        if token_record:
            break

    # Fallback: any token that matches the predicted body-part value.
    if token_record is None:
        for token_info in content_logprobs:
            if _token_matches_value(token_info.token, predicted_body_part):
                token_record = token_info
                break

    if token_record is None:
        return None

    return {
        "token": token_record.token,
        "logprob": token_record.logprob,
        "top_logprobs": [
            {"token": t.token, "logprob": t.logprob}
            for t in (token_record.top_logprobs or [])
        ],
    }