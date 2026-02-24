import re


def escape_features(expr, feature_names):
    for feat in sorted(feature_names, key=len, reverse=True):
        expr = re.sub(
            rf'(?<![A-Za-z0-9_`]){re.escape(feat)}(?![A-Za-z0-9_`])',
            f'`{feat}`',
            expr
        )
    return expr


def translate_lux_rule_to_query(rule_str, feature_names):
    match = re.search(r'IF (.+?) THEN', rule_str)
    if not match:
        return "()"

    condition_block = match.group(1)

    # Replace HTML entities
    condition_block = (condition_block
        .replace('&ge;', '>=')
        .replace('&le;', '<=')
        .replace('&gt;', '>')
        .replace('&lt;', '<')
        .replace('&ne;', '!=')
        .replace('&eq;', '==')
    )

    conditions = [cond.strip() for cond in condition_block.split('AND')]
    query_parts = []

    for cond in conditions:
        cond = cond.rstrip(') ').strip()

        match = re.match(
            r'(.+?)\s*(>=|<=|!=|==|>|<)\s*(.+)',
            cond
        )

        if match:
            lhs, operator, rhs = match.groups()

            # 🔹 ONLY NEW LOGIC
            lhs = escape_features(lhs, feature_names)
            rhs = escape_features(rhs, feature_names)

            query_parts.append(f"({lhs} {operator} {rhs})")
        else:
            query_parts.append(f"# Could not parse: {cond}")

    return "(" + " & ".join(query_parts) + ")"


import re
import numpy as np

def scale_single_value(scaler, feature_index, value):
    """
    Apply per-feature scaling manually.
    Works for StandardScaler and MinMaxScaler.
    """

    if hasattr(scaler, "mean_"):  # StandardScaler
        return (value - scaler.mean_[feature_index]) / scaler.scale_[feature_index]

    elif hasattr(scaler, "data_min_"):  # MinMaxScaler
        data_min = scaler.data_min_[feature_index]
        data_max = scaler.data_max_[feature_index]
        return (value - data_min) / (data_max - data_min)

    else:
        raise ValueError("Unsupported scaler type")


def encode_rules_with_scaler_and_le(cc_edu_rules, scaler, le, feature_order):
    """
    feature_order: list of feature names in the same order as training data (len = n_features)
    """

    encoded = []

    for rule_outer in cc_edu_rules:
        block = rule_outer[0][0]
        rule_dict = block["rule"]
        prediction = block["prediction"]

        new_rule = {}

        for feature, cond_list in rule_dict.items():

            feature_idx = feature_order.index(feature)
            new_conditions = []

            for cond in cond_list:
                m = re.match(r'([<>]=?|==)\s*([0-9.]+)', cond)
                if m:
                    op, val = m.groups()
                    val = float(val)

                    scaled_val = scale_single_value(scaler, feature_idx, val)
                    new_conditions.append(f"{op}{scaled_val}")

                else:
                    new_conditions.append(cond)  # ranges etc.

            new_rule[feature] = new_conditions

        encoded_pred = le.transform([prediction])[0]

        encoded.append([[
            {
                "rule": new_rule,
                "prediction": str(encoded_pred),
                "confidence": block.get("confidence", 1.0),
                "protected":1
            }
        ]])

    return encoded
