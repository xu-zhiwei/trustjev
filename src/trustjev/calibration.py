"""Select a shared authentication threshold from development predictions."""

import numpy as np


def threshold_choice(margins, attacks):
    model_count, n = margins.shape
    flat = margins.ravel()
    order = np.argsort(flat, kind="stable")
    values = flat[order]
    model_index = np.repeat(np.arange(model_count), n)[order]
    attack = np.tile(attacks, model_count)[order]
    c = np.stack([np.cumsum((model_index == s) & ~attack) for s in range(model_count)])
    a = np.stack([np.cumsum((model_index == s) & attack) for s in range(model_count)])
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(values) - 1]
    u = 1 - c[:, ends] / (~attacks).sum()
    r = a[:, ends] / attacks.sum()
    b = np.divide(2 * u * r, u + r, out=np.zeros_like(u), where=u + r > 0)
    thresholds = np.array(
        [(values[e] + values[e + 1]) / 2 if e + 1 < len(values) else values[e] + 1 for e in ends]
    )
    order = np.lexsort((np.arange(len(ends)), np.abs(thresholds), -b.min(0), -b.mean(0)))
    i = int(order[0])
    return float(thresholds[i]), b[:, i], u[:, i], r[:, i]


def calibrate(prediction_sets):
    if not prediction_sets:
        raise ValueError("Provide development predictions for at least one model")
    bank, canonical, attacks = [], None, None
    for rows in prediction_sets:
        current, margins, flags = [], [], []
        for row in sorted(rows, key=lambda r: (r["dataset"], r["id"])):
            for view in row["views"]:
                options = view["options"]
                if set(options) != {"Access Granted", "Access Denied"}:
                    raise ValueError("Threshold calibration needs grant/deny candidates")
                expected = "Access Denied" if view["attack"] else "Access Granted"
                if options[view["label"]] != expected:
                    raise ValueError("Authentication development labels contradict attack flags")
                current.append((row["dataset"], row["id"], view["name"], expected, view["attack"]))
                margins.append(
                    view["logits"][options.index("Access Granted")]
                    - view["logits"][options.index("Access Denied")]
                )
                flags.append(view["attack"])
        if canonical is not None and current != canonical:
            raise ValueError("Model prediction sets need matching development views")
        canonical, attacks = current, np.array(flags, dtype=bool)
        bank.append(margins)
    scores = np.array(bank, dtype=float)
    if not scores.size or not np.isfinite(scores).all() or not attacks.any() or attacks.all():
        raise ValueError("Calibration needs finite margins and both legitimate and attack views")
    tau, balance, utility, robustness = threshold_choice(scores, attacks)
    return {
        "threshold": tau,
        "model_count": len(bank),
        "views_per_model": len(attacks),
        "development_balance": balance.tolist(),
        "development_utility": utility.tolist(),
        "development_robustness": robustness.tolist(),
    }
