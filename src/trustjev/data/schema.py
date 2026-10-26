"""Validate labeled groups before tokenization or attack preparation."""


def validate_request(view, labeled=False):
    if not isinstance(view, dict):
        raise ValueError("Each view needs to be an object")
    for key in ("policy", "evidence"):
        if not isinstance(view.get(key), str):
            raise ValueError(f"{key} needs to be text")
    options = view.get("options")
    if (
        not isinstance(options, list)
        or len(options) < 2
        or not all(isinstance(x, str) and x for x in options)
        or len(set(options)) != len(options)
    ):
        raise ValueError("Provide at least two distinct nonempty candidate strings")
    if labeled:
        if not isinstance(view.get("name"), str) or not view["name"]:
            raise ValueError("Each labeled view needs a name")
        label = view.get("label")
        if type(label) is not int or not 0 <= label < len(options):
            raise ValueError("label needs to index a valid candidate")
        if any(type(view.get(key)) is not bool for key in ("attack", "requires_evidence")):
            raise ValueError("attack and requires_evidence need to be booleans")


def validate_group(record, authentication=False):
    for key in ("id", "dataset", "task"):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f"Each group needs a nonempty {key}")
    if not isinstance(record.get("views"), list) or not record["views"]:
        raise ValueError("Each group needs a nonempty views list")
    names = {}
    options = None
    for view in record["views"]:
        validate_request(view, labeled=True)
        if view["name"] in names:
            raise ValueError("View names need to be unique within a group")
        names[view["name"]] = view
        if options is not None and view["options"] != options:
            raise ValueError("Candidate order needs to agree across views in a group")
        options = view["options"]
    if record.get("pairing", "explicit") not in {"explicit", "interventions"}:
        raise ValueError("pairing needs to be explicit or interventions")
    if type(record.get("adversarial", False)) is not bool:
        raise ValueError("adversarial needs to be a boolean")
    if record.get("pairing", "explicit") == "interventions":
        required = {
            "clean",
            "injected",
            "fact_change",
            "policy_change",
            "trusted_override",
            "untrusted_override",
        }
        if not required <= names.keys():
            raise ValueError("Intervention groups need the six named views")
        if ("operator_clean_or" in names) != ("operator_injected_or" in names):
            raise ValueError("Operator augmentation needs both endpoints")
    for key, same in (
        ("verified_invariance_pairs", True),
        ("verified_counterfactual_pairs", False),
    ):
        pairs = record.get(key, [])
        if not isinstance(pairs, list):
            raise ValueError(f"{key} needs to be a list")
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2 or any(x not in names for x in pair):
                raise ValueError("A checked pair needs two existing view names")
            if (names[pair[0]]["label"] == names[pair[1]]["label"]) != same:
                raise ValueError("Checked pair labels contradict the declared relation")
    if authentication:
        if len(names) != 2 or set(options) != {"Access Granted", "Access Denied"}:
            raise ValueError("Authentication needs two views and grant/deny candidates")
        if {v["attack"] for v in names.values()} != {False, True}:
            raise ValueError("Authentication needs one legitimate view and one attack")
        for view in names.values():
            target = "Access Denied" if view["attack"] else "Access Granted"
            if view["options"][view["label"]] != target:
                raise ValueError("Authentication labels contradict legitimate/attack roles")
