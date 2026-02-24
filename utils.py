from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    make_classification,
    make_moons,
    make_circles

)
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import pandas as pd

from collections import defaultdict
from typing import Tuple, Dict, Optional
import openml
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from lux.lux import LUX
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import f1_score

def load_dataset_oml(name: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load an OpenML dataset by a friendly name and label-encode all categorical features.
    The target y is also label-encoded if it is non-numeric.

    Parameters
    ----------
    name : str
        One of the keys defined in `dataset_ids`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with categorical columns label-encoded (dtype: int).
    y : np.ndarray
        Label-encoded target (if originally categorical) or numeric target as-is.

    Raises
    ------
    ValueError
        If `name` is unknown or mapped to None (e.g., UCI dataset without OpenML ID).
    """
    name = name.lower()

    # numerical-only OpenML datasets (with IDs)
    dataset_ids: Dict[str, Optional[int]] = {
        "openml_sonar": 40,
        "openml_credit_g": 31,        # 21 features, 1000 instances (borderline)
    }

    if name not in dataset_ids:
        raise ValueError(f"Unknown dataset: {name}")

    did = dataset_ids[name]
    if did is None:
        raise ValueError(
            f"Dataset '{name}' does not have an OpenML ID in this mapping. "
            "Please provide a valid OpenML ID or use an OpenML-available dataset key."
        )

    # Fetch dataset metadata and data
    data = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, feature_names = data.get_data(
        dataset_format="dataframe",
        target=data.default_target_attribute
    )

    # --- Label-encode categorical feature columns ---
    # Fallback heuristic: also treat object/category dtypes as categorical, in case indicator is incomplete
    cat_cols_from_indicator = [fn for fn, is_cat in zip(feature_names, categorical_indicator) if is_cat]
    cat_cols_from_dtype = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
    cat_cols = list(dict.fromkeys(cat_cols_from_indicator + cat_cols_from_dtype))  # unique, stable order

    encoders_X: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Convert to string and fill NA with a sentinel to make encoding stable
        col_values = X[col].astype("string").fillna("<NA>").to_numpy()
        X[col] = le.fit_transform(col_values).astype(int)
        encoders_X[col] = le

    # --- Label-encode the target if needed ---
    y_encoded = y
    if isinstance(y, (pd.Series, pd.Categorical)) or hasattr(y, "dtype"):
        y_is_numeric = pd.api.types.is_numeric_dtype(y)
        if not y_is_numeric:
            le_y = LabelEncoder()
            # cast to string, handle NA similarly
            y_encoded = le_y.fit_transform(pd.Series(y, dtype="string").fillna("<NA>"))
        else:
            # Ensure numpy array
            y_encoded = np.asarray(y)
    else:
        # Non-pandas sequence: try numeric detection; if not numeric, encode as strings
        try:
            y_arr = np.asarray(y, dtype=float)
            y_encoded = y_arr
        except Exception:
            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(pd.Series(y, dtype="string").fillna("<NA>"))

    # If we need later need the encoders (for inverse_transform), you can return them as well:
    # return X, y_encoded, encoders_X, le_y if not y_is_numeric else None

    return X, y_encoded


def load_classification_dataset(
    name="iris",
    n_samples=100,
    n_features=20,
    n_classes=2,
    noise=0.1,
    random_state=42
):
    """
    Load a classification dataset as DataFrame (X) and array (y).

    Parameters:
    -----------
    name : str
        Dataset name. Options:
            - 'iris'
            - 'wine'
            - 'breast_cancer'
            - 'moons'
            - 'circles'
            - 'synthetic'
    n_samples : int
        Used for synthetic, two_moons, circles
    n_features : int
        Only used for 'synthetic'
    n_classes : int
        Only used for 'synthetic'
    noise : float
        Noise level for two_moons and circles
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X_df : pd.DataFrame
        Features
    y : np.ndarray
        Target labels
    """

    if name == "iris":
        data = load_iris()
        X_df = pd.DataFrame(
            data.data,
            columns=[f.replace("(cm)", "").replace(" ", "_") for f in data.feature_names]
        )
        y = data.target

    elif name == "wine":
        data = load_wine()
        X_df = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

    elif name == "breast_cancer":
        data = load_breast_cancer()
        X_df = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target


    elif name == "synthetic":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.6),
            n_redundant=int(n_features * 0.2),
            n_repeated=0,
            n_classes=n_classes,
            random_state=random_state
        )
        X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    elif name == "pima":
        data = fetch_openml(
            name="diabetes",
            version=1,
            as_frame=True
        )
        X_df = data.data
        y = data.target.map({'tested_negative': 0, 'tested_positive': 1}).to_numpy()
    elif 'openml' in name:
        
        X_df,y = load_dataset_oml(name)
        la = LabelEncoder()
        y = la.fit_transform(y)
    elif name == "coalmine":
        #takes long time, skip for now
        coal=pd.read_csv("https://gitlab.geist.re/pml/x_benchmark-with-selected-datasets/-/raw/main/CDS1/clustering-results-coal-mine-knac.zip")
        coal.columns = ['timestamp', 'cluster',
         'LCD_AverageThree_phaseCurrent_discrete',
               'RCD_AverageThree_phaseCurrent_discrete', 'LHD_EngineCurrent_discrete',
               'RHD_EngineCurrent_discrete', 'LP_AverageThree_phaseCurrent_discrete',
               'SM_ShearerSpeed_discrete',
               'LHD_LeftHaulageDrive_tractor_Temperature_gearbox_discrete',
               'LA_LeftArmTemperature_discrete', 'SM_DailyRouteOfTheShearer_discrete',
               'SM_TotalRoute_discrete', 'SM_ShearerLocation_discrete',
               'SM_ShearerMoveInLeft', 'SM_ShearerMoveInRight',
               'RCD_AverageThree_phaseCurrent', 'LCD_AverageThree_phaseCurrent',
               'LP_AverageThree_phaseCurrent', 'LHD_EngineCurrent',
               'RHD_EngineCurrent', 'SM_ShearerSpeed',
               'LHD_LeftHaulageDrive_tractor_Temperature_gearbox_',
               'LA_LeftArmTemperature', 'SM_DailyRouteOfTheShearer', 'SM_TotalRoute',
               'SM_ShearerLocation', 'expert']
        coal.sort_values(by='timestamp', inplace=True)
        features= [f for f in coal.columns if f not in ['expert','timestamp','cluster']]
        target = 'expert'
        X_df=coal[features]
        y = coal[target]
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    return X_df, y

def conditions_to_lux_rule(conditions):
    """
    Convert [('feat', '>=', thr), ...] to:
    {'feat': ['>=thr'], ...}
    """
    lux_rule = {}

    for feat, op, thr in conditions:
        if feat not in lux_rule:
            lux_rule[feat] = []
        lux_rule[feat].append(f"{op}{thr}")

    return lux_rule

def normalized_rule_to_lux(rule):
    """
    Convert internal rule format back to LUX-compatible dict.
    """
    return {
        'rule': conditions_to_lux_rule(rule['conditions']),
        'prediction': rule['prediction'],
        'confidence': rule['confidence'],
    }


def parse_lux_rule(lux_justify_output):
    """
    Parse a raw LUX explanation (justify output) into a normalized
    internal rule structure used throughout the CC‑EDU framework.

    LUX returns:
        [[{'rule': {...}, 'prediction': y_hat, 'confidence': c}]]

    This function extracts:
        - prediction label
        - rule confidence
        - a list of atomic conditions expressed as tuples:
              (feature, operator, threshold_or_expression)

    These conditions collectively define the intensional form of
    the rule; during CC‑EDU computation, these are mapped to the
    rule's *extensional* coverage set C_i (Definition 1).
    """

    if not lux_justify_output or not isinstance(lux_justify_output, list) or len(lux_justify_output) == 0:
        return None
    block = lux_justify_output[0]
    if not isinstance(block, list) or len(block) == 0:
        return None
    item = block[0]
    
    rule_dict = {
        'conditions': [],
        'prediction': item.get('prediction'),
        'confidence': float(item.get('confidence', 0.0))
    }
    
    raw_rules = item.get('rule', {})
    for feat_expr, cond_list in raw_rules.items():
        for cond_str in cond_list:
            match_simple = re.match(r'([<>]=?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', cond_str.strip())
            match_oblique = re.match(r'([<>]=?)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\*\s*(\w+)\s*([+-])\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', cond_str.strip())
            
            if match_simple:
                op, thresh = match_simple.groups()
                rule_dict['conditions'].append((feat_expr, op, float(thresh)))
            elif match_oblique:
                op, coef, feat, sign, const = match_oblique.groups()
                coef = float(coef)
                const_val = float(const) if sign == '+' else -float(const)
                rule_dict['conditions'].append((feat_expr, op, f"{coef} * {feat} + {const_val}"))
            else:
                rule_dict['conditions'].append((feat_expr, None, cond_str))
    
    return rule_dict

def rule_to_query(rule):

    """
    Convert parsed rule into a pandas query string, allowing us to compute
    its empirical coverage set C_i.

    According to how the paper operationalizes:
          C_i = { x in X : x satisfies R_i }

    Only axis-aligned conditions are translated here; oblique ones pass
    through as raw strings and may require manual adaptation depending
    on dataset column naming.
    """

    cond_strs = []
    for cond in rule['conditions']:
        if isinstance(cond, tuple):
            feat, op, thresh = cond
            # Clean op to pandas syntax
            if op == '>=':
                op_q = '>='
            elif op == '>':
                op_q = '>'
            elif op == '<=':
                op_q = '<='
            elif op == '<':
                op_q = '<'
            else:
                continue  # skip unknown
            cond_strs.append(f"`{feat}` {op_q} {thresh}")
        elif isinstance(cond, str):
            # Raw string: assume like '>= -1.268 * petal_length_ + 7.799'
            # Replace to pandas-safe (e.g., add backticks if needed, but assume feats are valid col names)
            cond_strs.append(cond)  # May need cleaning for oblique
    return ' & '.join(cond_strs) if cond_strs else 'True'  # Default true if no conds


import numpy as np

def rule_bounds(rule):
    """
    Extract per-feature bounds from rule conditions.
    Returns: {feat: {'lb': value or -inf, 'ub': value or +inf}}
    """
    bounds = {}

    for feat, op, thr in rule['conditions']:
        if feat not in bounds:
            bounds[feat] = {'lb': -np.inf, 'ub': np.inf}

        if op == '>=':
            bounds[feat]['lb'] = max(bounds[feat]['lb'], thr)
        elif op == '>':
            bounds[feat]['lb'] = max(bounds[feat]['lb'], thr)
        elif op == '<=':
            bounds[feat]['ub'] = min(bounds[feat]['ub'], thr)
        elif op == '<':
            bounds[feat]['ub'] = min(bounds[feat]['ub'], thr)

    return bounds


def satisfies_rule_row(rule_dict, row):
    """
    Check if a single row satisfies a rule.
    rule_dict example:
      {'feature1': (low, high), 'feature2': ('<=', 0.4)}
    """
    for feat, cond in rule_dict.items():
        val = row[feat]

        if isinstance(cond, tuple) and len(cond) == 2:
            lo, hi = cond
            if not (lo <= val <= hi):
                return False

        elif isinstance(cond, tuple) and len(cond) == 2 and isinstance(cond[0], str):
            op, thr = cond
            if op == "<=" and not val <= thr:
                return False
            if op == ">=" and not val >= thr:
                return False

        else:
            raise ValueError(f"Unknown condition format: {cond}")

    return True


from itertools import combinations

def satisfies_rule(rule_dict, feat_x, feat_y, x_val, y_val):
    satisfied = True
    for feat, cond_list in rule_dict.items():
        if feat != feat_x and feat != feat_y:
            continue  # Ignore other features
        
        for cond_str in cond_list:
            cond_str = cond_str.strip()
            if cond_str.startswith(('>=', '>', '<=', '<')):
                op = cond_str[:2] if cond_str[:2] in ['>=', '<='] else cond_str[0]
                thresh_str = cond_str[len(op):].strip()
                try:
                    thresh = float(thresh_str)
                    feat_val = x_val if feat == feat_x else y_val
                    if op == '>=':
                        satisfied &= (feat_val >= thresh)
                    elif op == '>':
                        satisfied &= (feat_val > thresh)
                    elif op == '<=':
                        satisfied &= (feat_val <= thresh)
                    elif op == '<':
                        satisfied &= (feat_val < thresh)
                except ValueError:
                    satisfied = False
                    break
            else:
                satisfied = False
                break
    return satisfied


def get_covered_indices(df, rule):

    """
    Compute the empirical coverage set C_i for rule R_i.

    In the paper:
        C_i = indices of dataset X satisfying intensional rule conditions.

    This set forms the foundation for:
        - contradiction detection (Def. 1)
        - structural disagreement (Eq. 2)
      
    """
    query_str = rule_to_query(rule)
    if not query_str or query_str == 'True':
        return set(df.index)  # Covers all if no conditions
    try:
        covered_df = df.query(query_str)
        return set(covered_df.index)
    except Exception as e:
        print(f"Query error for rule: {query_str} - {e}")
        return set()  # Empty if error



def predict_with_lux_rules(X, rules_total_exp, default_class=None):
    """
    Predict labels for X using LUX rules in the nested format:
    rules_total_exp = [[[{'rule': {...}, 'prediction': '1', 'confidence': 0.6}]], ...]

    Returns:
        y_pred: np.array of predicted labels
        coverage: np.array of fraction of rules covering each instance
    """
    # Flatten rules inside
    flat_rules = []
    for item in rules_total_exp:
        # item is a list of lists
        for sublist in item:
            for rule_dict in sublist:
                flat_rules.append({
                    "rule_dict": rule_dict.get("rule", {}),
                    "prediction": rule_dict.get("prediction"),
                    "confidence": float(rule_dict.get("confidence", 1.0))
                })
    
    n_samples = X.shape[0]
    y_pred = np.empty(n_samples, dtype=object)
    coverage = np.zeros(n_samples)  # how many rules fired
    
    def satisfies_rule_row(rule_dict, row):
        """Check if a single row satisfies a rule (empty rule covers all)."""
        if not rule_dict:
            return True
        for feat, cond_list in rule_dict.items():
            for cond_str in cond_list:
                cond_str = cond_str.strip()
                if cond_str.startswith(('>=', '>', '<=', '<')):
                    op = cond_str[:2] if cond_str[:2] in ['>=', '<='] else cond_str[0]
                    try:
                        thresh = float(cond_str[len(op):])
                        val = row[feat]
                        if op == '>=' and not val >= thresh:
                            return False
                        elif op == '>' and not val > thresh:
                            return False
                        elif op == '<=' and not val <= thresh:
                            return False
                        elif op == '<' and not val < thresh:
                            return False
                    except:
                        return False
                else:
                    return False
        return True

    for idx in range(n_samples):
        row = X.iloc[idx]
        votes = defaultdict(float)
        votes_counts = defaultdict(float)
        rules_fired = 0
        
        
        for rule in flat_rules:
            if satisfies_rule_row(rule["rule_dict"], row):
                pred = rule["prediction"]
                votes[pred] += rule.get("confidence", 1.0)
                votes_counts[pred] +=1
                rules_fired += 1
        
        coverage[idx] = rules_fired / len(flat_rules) if flat_rules else 0.0
        
        if votes:
            for pred in votes.keys():
                votes[pred] = votes[pred]/votes_counts[pred] if votes_counts[pred]> 0 else 0
            # Choose the class with highest weighted votes
            y_pred[idx] = max(votes.items(), key=lambda x: x[1])[0]
        else:
            # Fallback
            y_pred[idx] = default_class if default_class is not None else None

    return np.array(y_pred), coverage

def plot_rule_parcoords(
    rules_parsed,
    edu_scores,
    X_test_scaled_df,
    class_colors=None,
    savefile='',
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    import numpy as np

    # ─────────────────────────────────────────────────────────────
    # Clean white scientific style
    # ─────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.size": 9
    })

    features = list(X_test_scaled_df.columns)
    n_feats = len(features)

    feat_min = X_test_scaled_df.min()
    feat_max = X_test_scaled_df.max()

    def scale_val(feat, val):
        lo, hi = feat_min[feat], feat_max[feat]
        if hi == lo:
            return 0.5
        return (val - lo) / (hi - lo)

    def unscale_val(feat, val_scaled):
        lo, hi = feat_min[feat], feat_max[feat]
        return lo + val_scaled * (hi - lo)

    # ─────────────────────────────────────────────────────────────
    # Class colors
    # ─────────────────────────────────────────────────────────────
    if class_colors is None:
        palette = plt.cm.tab10.colors
        all_preds = [r['prediction'] for r in rules_parsed if r is not None]
        unique_preds = sorted(set(all_preds))
        class_colors = {p: palette[i % 10] for i, p in enumerate(unique_preds)}

    norm_edu = Normalize(vmin=0, vmax=max(edu_scores.max(), 0.01))

    # ─────────────────────────────────────────────────────────────
    # Figure
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(12, n_feats * 2.2), 7),
        facecolor="white"
    )
    ax.set_facecolor("white")

    tick_positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    # ─────────────────────────────────────────────────────────────
    # Vertical axes
    # ─────────────────────────────────────────────────────────────
    for fi, feat in enumerate(features):
        ax.axvline(fi, color='#cccccc', linewidth=1.0, zorder=1)

        ax.text(
            fi, -0.08, feat,
            ha='center', va='top',
            fontsize=9,
            rotation=25,
            transform=ax.get_xaxis_transform()
        )

        for tick_pos in tick_positions:
            real_val = unscale_val(feat, tick_pos)

            ax.plot(
                [fi - 0.04, fi + 0.04],
                [tick_pos, tick_pos],
                color='#bbbbbb',
                linewidth=0.8,
                zorder=5
            )

            ax.text(
                fi - 0.06, tick_pos,
                f"{real_val:.2f}",
                ha='right',
                va='center',
                fontsize=6.5,
                zorder=5
            )

    ax.set_xlim(-0.5, n_feats - 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ─────────────────────────────────────────────────────────────
    # Draw rules
    # ─────────────────────────────────────────────────────────────
    for i, rule in enumerate(rules_parsed):

        if rule is None:
            continue

        edu = edu_scores[i]
        alpha = norm_edu(edu)

        pred = rule['prediction']
        color = class_colors.get(pred, 'black')

        bounds = rule_bounds(rule)

        xs, y_lows, y_highs, is_constrained = [], [], [], []

        for feat in features:
            b = bounds.get(feat, {})
            lb = b.get('lb', -np.inf)
            ub = b.get('ub', np.inf)

            feat_constrained = not (np.isinf(lb) and np.isinf(ub))
            is_constrained.append(feat_constrained)

            lb_plot = scale_val(feat, max(lb, feat_min[feat])) if not np.isinf(lb) else 0.0
            ub_plot = scale_val(feat, min(ub, feat_max[feat])) if not np.isinf(ub) else 1.0

            xs.append(len(xs))
            y_lows.append(lb_plot)
            y_highs.append(ub_plot)

        xs = np.array(xs)
        y_lows = np.array(y_lows)
        y_highs = np.array(y_highs)
        y_mids = (y_lows + y_highs) / 2

        # ── Segment drawing ─────────────────────────────
        for fi in range(len(xs) - 1):

            left_constrained = is_constrained[fi]
            right_constrained = is_constrained[fi + 1]

            poly_x = [xs[fi], xs[fi+1], xs[fi+1], xs[fi]]
            poly_y = [y_lows[fi], y_lows[fi+1], y_highs[fi+1], y_highs[fi]]

            if left_constrained and right_constrained:

                ax.fill(
                    poly_x, poly_y,
                    color=color,
                    alpha=alpha * 0.45,
                    zorder=2
                )

            elif left_constrained or right_constrained:

                ax.fill(
                    poly_x, poly_y,
                    color=color,
                    alpha=alpha * 0.15,
                    zorder=2,
                    hatch='////',
                    linewidth=0
                )

                constrained_fi = fi if left_constrained else fi + 1

                ax.plot(
                    [xs[constrained_fi], xs[constrained_fi]],
                    [y_lows[constrained_fi], y_highs[constrained_fi]],
                    color=color,
                    alpha=alpha,
                    linewidth=3.5,
                    solid_capstyle='round',
                    zorder=4
                )

            else:
                ax.fill(
                    poly_x, poly_y,
                    color='#dddddd',
                    alpha=0.15,
                    zorder=1
                )

        # ── Constrained axis markers ─────────────────────
        for fi in range(len(xs)):
            if is_constrained[fi]:
                ax.plot(
                    [xs[fi], xs[fi]],
                    [y_lows[fi], y_highs[fi]],
                    color=color,
                    alpha=alpha,
                    linewidth=4.0,
                    solid_capstyle='round',
                    zorder=4
                )

        # ── Centerline ───────────────────────────────────
        constrained_xs = xs[np.array(is_constrained)]
        constrained_mids = y_mids[np.array(is_constrained)]

        if len(constrained_xs) > 0:
            ax.plot(
                constrained_xs,
                constrained_mids,
                color=color,
                alpha=alpha,
                linewidth=1.2 + 1.5 * norm_edu(edu),
                zorder=5,
                marker='o',
                markersize=3
            )

    # ─────────────────────────────────────────────────────────────
    # Legend
    # ─────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=c, label=f'Class {p}')
        for p, c in class_colors.items()
    ]

    constrained_note = mpatches.Patch(
        facecolor='gray',
        alpha=0.4,
        label='Solid band = both axes constrained'
    )

    partial_note = mpatches.Patch(
        facecolor='gray',
        alpha=0.15,
        hatch='////',
        label='Hatched = one axis unconstrained'
    )

    edu_note = mpatches.Patch(
        facecolor='gray',
        alpha=0.3,
        label='Opacity + line weight ~ CC-EDU'
    )

    ax.legend(
        handles=patches + [constrained_note, partial_note, edu_note],
        loc='upper right',
        fontsize=8,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc'
    )

    ax.set_title(
        'Rule-Region Parallel Coordinates — Feature Bounds per Rule\n'
        'Bold bar = constrained axis  |  Hatched = extends to boundary  |  '
        'Opacity = CC-EDU',
        fontsize=12,
        pad=18
    )

    plt.tight_layout()

    plt.savefig(
        f"parcoords_rules_{savefile}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.show()


import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any

def save_grid_results(base_dir: str,
                      dnsname: str,
                      dsname: str,
                      edu_avg_grid: np.ndarray,
                      f1_all_grid: np.ndarray,
                      edu_avg_grid_r: np.ndarray,
                      f1_all_grid_r: np.ndarray,
                      extra_meta:  Optional[Dict[str, Any]] = None) -> str:
    """
    Save grid-search result arrays and metadata under:
        {base_dir}/{dnsname}/{dsname}.npz and {dsname}.json

    Returns the path to the saved .npz file.
    """
    save_dir = os.path.join(base_dir, dnsname)
    os.makedirs(save_dir, exist_ok=True)

    npz_path = os.path.join(save_dir, f"{dsname}.npz")
    np.savez_compressed(
        npz_path,
        edu_avg_grid=edu_avg_grid,
        f1_all_grid=f1_all_grid,
        edu_avg_grid_r=edu_avg_grid_r,
        f1_all_grid_r=f1_all_grid_r,
    )

    meta = {
        "dnsname": dnsname,
        "dsname": dsname,
        "shape": {
            "edu_avg_grid": list(edu_avg_grid.shape),
            "f1_all_grid": list(f1_all_grid.shape),
            "edu_avg_grid_r": list(edu_avg_grid_r.shape),
            "f1_all_grid_r": list(f1_all_grid_r.shape),
        },
        "saved_at": datetime.now().isoformat(timespec="seconds")
    }
    if extra_meta:
        meta.update(extra_meta)

    json_path = os.path.join(save_dir, f"{dsname}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return npz_path