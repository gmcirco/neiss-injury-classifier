from pydantic import BaseModel, Field
from enum import Enum

class BodyPart(str, Enum):
    INTERNAL = "0"
    SHOULDER = "30"
    UPPER_TRUNK = "31"
    ELBOW = "32"
    LOWER_ARM = "33"
    WRIST = "34"
    KNEE = "35"
    LOWER_LEG = "36"
    ANKLE = "37"
    PUBIC_REGION = "38"
    HEAD = "75"
    FACE = "76"
    EYEBALL = "77"
    LOWER_TRUNK = "79"
    UPPER_ARM = "80"
    UPPER_LEG = "81"
    HAND = "82"
    FOOT = "83"
    BODY_25_50_PERCENT = "84"
    ALL_PARTS_BODY = "85"
    NOT_STATED_UNKNOWN = "87"
    MOUTH = "88"
    NECK = "89"
    FINGER = "92"
    TOE = "93"
    EAR = "94"


class InjuryClassification(BaseModel):
    body_part: BodyPart = Field(
        description="ID code for the primary body part injured or involved in the narrative"
    )
    reasoning: str = Field(
        description="10 word or less description of why the body part was chosen"
    )
    confidence: float = Field(
        description=(
            "Confidence of the body part classification as a float from 0.0 to 1.0, "
            "where 1.0 is absolute certainty, and 0.0 is completely unknown"
        )
    )