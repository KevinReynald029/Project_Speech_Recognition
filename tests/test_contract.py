import json
import unittest
from pathlib import Path

try:
    import jsonschema
except ImportError:
    jsonschema = None


@unittest.skipIf(jsonschema is None, "Install .[test] to validate the JSON schemas")
class ContractTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.schema = json.loads((self.root/"schemas"/"result.schema.json").read_text())

    def test_success_and_all_failure_fixtures_match_schema(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)
        for path in (self.root/"examples"/"generated").glob("*.json"):
            if path.name != "reference.json":
                with self.subTest(file=path.name):
                    jsonschema.validate(json.loads(path.read_text()), self.schema)

    def test_failure_cannot_be_misrepresented_as_numeric_score(self):
        response = json.loads((self.root/"examples"/"generated"/"no_speech.json").read_text())
        response["phoneme_accuracy_summary"] = 0
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(response, self.schema)

    def test_reference_fixture_matches_schema(self):
        schema = json.loads((self.root/"schemas"/"reference.schema.json").read_text())
        jsonschema.validate(json.loads((self.root/"examples"/"reference.json").read_text()), schema)
