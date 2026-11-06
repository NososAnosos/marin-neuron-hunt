"""Checks every explanation and request file against its schema and the repo's naming rules.
examples/ is checked against the schemas too, but not the location rules.

Usage: python scripts/validate.py
"""

import json
import re
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parent.parent
REQUEST_ID = re.compile(r"^(L\d+-N\d+)-(\d{3})$")


def load_schemas():
    schemas = {name: json.loads((ROOT / "schema" / f"{name}.schema.json").read_text())
               for name in ("explanation", "request", "result")}
    registry = Registry().with_resources(
        (f"{name}.schema.json", Resource.from_contents(schema)) for name, schema in schemas.items()
    )
    return {name: Draft202012Validator(schema, registry=registry) for name, schema in schemas.items()}


def load_yaml(path, errors):
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as err:
        errors.append(f"{path}: not valid YAML ({err})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{path}: must be a YAML mapping")
        return None
    return data


def schema_errors(validator, data, path, errors):
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        where = "/".join(str(p) for p in err.path) or "(top level)"
        errors.append(f"{path}: {where}: {err.message}")


def check_explanation(path, data, validator, errors, example=False):
    schema_errors(validator, data, path, errors)
    neuron = data.get("neuron")
    if isinstance(neuron, str) and not example:
        layer = neuron.split("-")[0]
        expected = ROOT / "explanations" / layer / f"{neuron}.yaml"
        if path != expected:
            errors.append(f"{path}: a file for {neuron} must live at {expected.relative_to(ROOT)}")
    label = data.get("label")
    if isinstance(label, str) and len(label.split()) > 8:
        errors.append(f"{path}: label has {len(label.split())} words; keep it to 8")
    requests_dir = ROOT / ("examples" if example else "requests")
    for alt in data.get("alternatives") or []:
        rid = alt.get("discriminating_request") if isinstance(alt, dict) else None
        if isinstance(rid, str) and not (requests_dir / f"{rid}.yaml").exists():
            errors.append(f"{path}: alternatives names request {rid}, but requests/{rid}.yaml doesn't exist")


def check_request(path, data, validator, errors):
    schema_errors(validator, data, path, errors)
    m = REQUEST_ID.match(path.stem)
    if not m:
        errors.append(f"{path}: name it <neuron>-<nnn>.yaml, e.g. L40-N25546-001.yaml")
        return
    named = data.get("neuron") or (data.get("neurons") or [None])[0]
    if named != m.group(1):
        errors.append(f"{path}: file name says {m.group(1)} but the request's (first) neuron is {named}")


def main():
    validators = load_schemas()
    errors = []
    count = 0
    for path in sorted((ROOT / "explanations").rglob("*")):
        if path.is_file() and path.suffix in (".yaml", ".yml"):
            count += 1
            if path.suffix == ".yml":
                errors.append(f"{path}: use the .yaml extension")
            data = load_yaml(path, errors)
            if data is not None:
                check_explanation(path, data, validators["explanation"], errors)
    for path in sorted((ROOT / "requests").glob("*.y*ml")):
        count += 1
        data = load_yaml(path, errors)
        if data is not None:
            check_request(path, data, validators["request"], errors)
    for path in sorted((ROOT / "results").glob("*.y*ml")):
        count += 1
        data = load_yaml(path, errors)
        if data is not None:
            schema_errors(validators["result"], data, path, errors)
            if data.get("request") != path.stem:
                errors.append(f"{path}: file name says {path.stem} but the result is for {data.get('request')}")
    for path in sorted((ROOT / "examples").glob("*.yaml")):
        count += 1
        data = load_yaml(path, errors)
        if data is None:
            continue
        if REQUEST_ID.match(path.stem):
            check_request(path, data, validators["request"], errors)
        else:
            check_explanation(path, data, validators["explanation"], errors, example=True)
    for message in errors:
        print(message)
    print(f"checked {count} files: {'OK' if not errors else f'{len(errors)} problem(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
