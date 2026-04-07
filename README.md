## NEISS Injury Classifier

Classifies NEISS injury narratives to the correct primary body part

### Project Structure

- Prompts are stored in `src/prompts.py`
- Pydantic extraction schema in `src/schema/py`
- General utility functions in `src/utils/py`

Two 'entrypoints' for running the models:

- `classify_narratives.py` runs locally, deposits results in artifacts/output
- `classify_narratives_batch.py` builds result, deposits batch input in artifacts/batch