from utils import *

def rules_disagree(rule_a, rule_b, df, overlap_threshold=0.5, alpha=0.3, beta=0.2):

    """
    Structural Disagreement score (Eq. 2 in the paper).

    disagree(R_i, R_j) =
        |(C_i ∩ C_j)| / |(C_i ∪ C_j)|   ×   (0.5 + αF_o + βConf_p)

    • Spatial overlap term (dominant component)
    • Feature overlap (shared feature constraints)
    • Confidence penalty (alignment bonus)

    Returned value ∈ [0, 1].
    """

    if not rule_a or not rule_b:
        return 0.0
    
    # Empirical coverage
    indices_a = get_covered_indices(df, rule_a)
    indices_b = get_covered_indices(df, rule_b)
    
    if not indices_a or not indices_b:
        return 0.0
    
    intersection = indices_a & indices_b
    union = indices_a | indices_b
    spatial_overlap = len(intersection) / len(union) if len(union) > 0 else 0.0
    
    if spatial_overlap < overlap_threshold:
        return 0.0
    
    # Optional: still include feat overlap bonus if needed
    feats_a = {c[0] for c in rule_a['conditions'] if c[0] is not None}
    feats_b = {c[0] for c in rule_b['conditions'] if c[0] is not None}
    shared = feats_a & feats_b
    union_feats = feats_a | feats_b
    feat_overlap = len(shared) / len(union_feats) if union_feats else 0.0
    
    conf_bonus = min(rule_a['confidence'], rule_b['confidence']) if rule_a['prediction'] == rule_b['prediction'] else 0.0

    # weighting is in fact not that important, it is more for visualizaiotn, as cc-edu is a sum
    score = spatial_overlap * (0.5 + alpha * feat_overlap + beta * conf_bonus)
    return min(max(score, 0.0), 1.0)


def is_contradictory(rule_a, rule_b, df, min_overlap=0.1, knn_indices=None, instance_idx=None):

    """
    Implementation of Definition 1: Neighborhood-conditioned contradiction.

    Two rules R_i, R_j contradict w.r.t instance x_i if:
        1) They predict different classes
        2) Their coverage sets sufficiently overlap *within* N_k(i)

    This corresponds exactly to Equation (1) from the paper.
    """

    if rule_a['prediction'] == rule_b['prediction']:
        return False

    indices_a = get_covered_indices(df, rule_a)
    indices_b = get_covered_indices(df, rule_b)

    if knn_indices is not None and instance_idx is not None:
        neighbors = set(knn_indices[instance_idx][1:])
        indices_a &= neighbors
        indices_b &= neighbors

    if not indices_a or not indices_b:
        return False

    intersection = indices_a & indices_b
    union = indices_a | indices_b
    spatial_overlap = len(intersection) / len(union) if union else 0.0

    return spatial_overlap >= min_overlap


def extract_rules_lux(
    X_test_df_scaled,
    X_train_scaled_df,
    y_train,
    mw,
    lux_locality=0.02,
):
    rules = []
    rules_parsed=[]

    lux = LUX(
        predict_proba=mw.predict_proba,
        neighborhood_size=lux_locality,
        max_depth=2,
        oversampling_strategy='importance',
        node_size_limit=2,
        grow_confidence_threshold=0
    )

    for i in range(len(X_test_df_scaled)):
        instance_df = X_test_df_scaled.iloc[[i]]
        instance_np = instance_df.to_numpy()

        lux.fit(
            X_train_scaled_df,
            y_train,
            instance_to_explain=instance_np,
            tree_with_shap=False,
            oblique=False
        )
        justify = lux.justify(instance_np, to_dict=True)
        rule = parse_lux_rule(justify)
        rules_parsed.append(rule)
        rules.append(justify)

    return rules, rules_parsed, lux


from sklearn.neighbors import NearestNeighbors
import numpy as np


def compute_cc_edu(
    rules,
    X_test_df_scaled,
    k=20,  # if None -> global cc-edu
    overlap_threshold=0.1,
    tqdm=False
):
    """
    Compute CC-EDU for each instance (Equation 3):

        CC-EDU(R_i) = (1 / |N_k(i)|) * Σ_{j ∈ N_k(i)}
                           contr_i(R_i, R_j) * disagree(R_i, R_j)

    • If k is None → global comparison.
    • Otherwise → kNN locality.
    • tqdm=True enables progress bar.
    """

    X_np = X_test_df_scaled.to_numpy()
    n = len(rules)

    # Compute kNN only if locality is used
    if k is not None:
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_np)
        _, knn_indices = nbrs.kneighbors(X_np)
    else:
        knn_indices = None

    edu_scores = np.zeros(n)

    # Optional tqdm wrapper
    iterator = range(n)
    if tqdm:
        from tqdm import tqdm as tqdm_lib
        iterator = tqdm_lib(iterator, desc="Computing CC-EDU")

    for i in iterator:
        rule_i = rules[i]
        if rule_i is None:
            continue

        disagreement = 0.0
        valid_count = 0

        # Define comparison set
        if k is None:
            neighbor_indices = (j for j in range(n) if j != i)
        else:
            neighbor_indices = knn_indices[i][1:]  # skip self

        for j in neighbor_indices:
            rule_j = rules[j]
            if rule_j is None:
                continue

            # Definition 1 (contradiction)
            if is_contradictory(
                rule_i,
                rule_j,
                X_test_df_scaled,
                min_overlap=overlap_threshold,
                knn_indices=knn_indices if knn_indices is not None else (np.arange(0,X_np.shape[0]),), #FIXME
                instance_idx=i if knn_indices is not None else 0 #FIXME
            ):
                # Eq. 2 (structural disagreement)
                disagreement += rules_disagree(
                    rule_i,
                    rule_j,
                    X_test_df_scaled,
                    overlap_threshold=overlap_threshold
                )

            valid_count += 1

        if valid_count > 0:
            edu_scores[i] = disagreement / valid_count

    return edu_scores, knn_indices

    

def compute_cc_edu_lux(X_train_df_scaled, y_train, X_test_df_scaled, mw, k=20, lux_locallity=0.02, overlap_threshold=0.1):
    rules,rules_parsed,lux = extract_rules_lux(
        X_test_df_scaled,
        X_train_df_scaled,
        y_train,
        mw,
        lux_locality=lux_locallity
        )
    
    
    edu_scores,knn_indices = compute_cc_edu(
        rules_parsed,
        X_test_df_scaled,
        k=k,
        overlap_threshold=overlap_threshold
    )
    
    return edu_scores, lux, rules,knn_indices

def directional_disagreement(rule_i, rule_j, df, knn_indices=None, instance_idx=None):

    """
    Compute directional disagreement strength on the region:

        D_ij = (C_i ∩ C_j) ∩ N_k(i)

    Used to determine whether rule_i is overgeneral and should be restricted.

    Strength = |D_ij| / |C_i|       (Equation 4)
    """

    Ci = get_covered_indices(df, rule_i)
    Cj = get_covered_indices(df, rule_j)

    if knn_indices is not None and instance_idx is not None:
        neighbors = set(knn_indices[instance_idx][1:])  # skip self
        Ci &= neighbors
        Cj &= neighbors

    if not Ci or not Cj:
        return 0.0, set()
    Dij = Ci & Cj
    strength = len(Dij) / len(Ci) if len(Ci) > 0 else 0.0
    return strength, Dij





from sklearn.tree import DecisionTreeClassifier

def learn_restriction_decision_tree(rule_i, disagree_idx, df, max_depth=1):
    """
    Learn a restriction from the disagreement region by fitting a small decision tree.
    Returns a list of new (feat, op, threshold) conditions.
    """
    covered_idx = get_covered_indices(df, rule_i)
    if not covered_idx or not disagree_idx:
        return None

    agree_idx = covered_idx - disagree_idx
    if not agree_idx:
        return None  # nothing to learn

    # Construct training data
    X = df.loc[list(covered_idx)].copy()
    y = np.array([1 if i in agree_idx else 0 for i in covered_idx])

    # Fit small decision tree
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X, y)

    # Convert tree splits into axis-aligned conditions
    conditions = []

    tree = clf.tree_
    features = X.columns

    # Only handle splits at root for simplicity (can extend for max_depth>1)
    def traverse(node_id=0):
        if tree.feature[node_id] == -2:  # leaf
            return

        feat = features[tree.feature[node_id]]
        thr = tree.threshold[node_id]

        # Determine which side is "agree"
        left_agree = tree.value[tree.children_left[node_id]][0][1]
        right_agree = tree.value[tree.children_right[node_id]][0][1]

        if left_agree > right_agree:
            # left ≤ threshold → agree
            conditions.append((feat, '<=', thr))
        else:
            # right > threshold → agree
            conditions.append((feat, '>=', thr))

        # traverse children
        traverse(tree.children_left[node_id])
        traverse(tree.children_right[node_id])

    traverse(0)
    return conditions if conditions else None




def restrict_rule(rule, restriction_cond):
    """
    Add a restriction condition to a rule (AND).
    """
    if restriction_cond is None:
        return rule

    new_rule = rule.copy()
    new_rule['conditions'] = list(rule['conditions']) + [restriction_cond]

    
    return new_rule


def restrict_rule_safe(rule_i, cond, instance_idx, df):
    """
    Only apply restriction if the explained instance still falls inside the rule.
    """
    candidate = restrict_rule(rule_i, cond)
    covered = get_covered_indices(df, candidate)
    if instance_idx is None or instance_idx in covered:
        return candidate
    return rule_i  # revert if instance got cut out



def restrict_rules(
    rules_total_exp,
    X_df,
    overlap_threshold=0.1,
    restriction_threshold=0.1,
    k=20,
    knn_indices=None,
    tqdm=False
):
    n = len(rules_total_exp)

    # Locality setup unchanged...
    if k is None and knn_indices is None:
        knn_indices = None
        X_np = X_df.to_numpy()
    elif knn_indices is None:
        X_np = X_df.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_np)
        _, knn_indices = nbrs.kneighbors(X_np)

    restricted_rules = []
    contradictions = []

    iterator = range(n)
    if tqdm:
        from tqdm import tqdm as tqdm_lib
        iterator = tqdm_lib(iterator, desc="Restricting rules")

    for i in iterator:

        rule_i_list = rules_total_exp[i]
        if not rule_i_list:
            restricted_rules.append([[]])
            print('index out of')
            continue

        rule_i = parse_lux_rule(rule_i_list)
        if rule_i is None:
            restricted_rules.append([[]])
            print('badrule')
            continue

        # --------------------------------------------------------
        # 1. PROTECTED — skip restriction completely
        # --------------------------------------------------------
        if rule_i.get("protected", 0) == 1:
            lux_rule = normalized_rule_to_lux(rule_i)
            restricted_rules.append([[lux_rule]])
            continue

        # neighbor handling unchanged...
        if knn_indices is None:
            neighbor_indices = (j for j in range(n) if j != i) 
        else:
            neighbor_indices = knn_indices[i][1:]

        for j in neighbor_indices:

            rule_j_list = rules_total_exp[j]
            if not rule_j_list:
                continue

            rule_j = parse_lux_rule(rule_j_list)
            if rule_j is None:
                continue

            # contradiction check unchanged
            if not is_contradictory(
                rule_i, rule_j, X_df,
                min_overlap=overlap_threshold,
                knn_indices=knn_indices if knn_indices is not None else (np.arange(0,X_np.shape[0]),), #FIXME
                instance_idx=i if knn_indices is not None else 0 #FIXME
            ):
                continue

            # overgeneral check
            Ci = get_covered_indices(X_df, rule_i)
            Cj = get_covered_indices(X_df, rule_j)

            rule_i_is_overgeneral = (
                len(Ci) > len(Cj) and
                rule_i["confidence"] <= rule_j["confidence"]
            )

            # --------------------------------------------------------
            # 2. If rule_j is PROTECTED → rule_i must be restricted
            # --------------------------------------------------------
            if rule_j.get("protected", 0) == 1:
                rule_i_is_overgeneral = True

            if not rule_i_is_overgeneral:
                continue

            # disagreement threshold unchanged
            strength, Dij = directional_disagreement(
                rule_i, rule_j, X_df, 
                knn_indices=knn_indices if knn_indices  is not None else (np.arange(0,X_np.shape[0]),), #FIXME
                instance_idx=i if knn_indices is not None else 0 #FIXME
            )

            if strength < restriction_threshold:
                continue

            contradictions.append([rule_i, rule_j])

            new_conds = learn_restriction_decision_tree(
                rule_i, Dij, X_df, max_depth=1
            )

            if new_conds:
                for cond in new_conds:
                    rule_i = restrict_rule_safe(rule_i, cond, i if knn_indices is not None else None, X_df) #no knn_indices means currently, global (no safe removal)

        lux_rule = normalized_rule_to_lux(rule_i)
        restricted_rules.append([[lux_rule]])

    return restricted_rules, knn_indices, contradictions
