"""
Amenity Tier feature engineering (Architecture C).

This module implements the design discussed in 03_Features.ipynb:
a portable concept dictionary (city-agnostic) combined with city-specific
Ridge weight learning. The output is a categorical `amenity_tier` in
{basic, premium, luxury}.

Public API
----------
- CONCEPTS                : the concept dictionary (portable across cities)
- build_concept_matrix    : turn a Series of amenity-list strings into a binary DataFrame
- fit_amenity_tier        : training-time, returns the tier + a reusable bundle
- transform_amenity_tier  : inference-time, applies a fitted bundle to new data

Bundle format
-------------
A `bundle` is a dict returned by `fit_amenity_tier` and consumed by
`transform_amenity_tier`. Keys:
    - 'model'                 : sklearn.linear_model.Ridge (fitted)
    - 'discriminant_concepts' : list[str] of concept names used by the model
    - 'tercile_thresholds'    : np.ndarray of shape (2,)
    - 'alpha'                 : float, the regularization strength used
    - 'prevalence_threshold'  : float, the prevalence cutoff used at fit time
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict


# ---------------------------------------------------------------------------
# Concept dictionary (portable across cities)
# ---------------------------------------------------------------------------

CONCEPTS = {
    # Basic comfort
    'wifi':              {'include': ['wifi', 'ethernet']},
    'heating':           {'include': ['heating']},
    'hot_water':         {'include': ['hot water'], 'exclude': ['hot water kettle']},
    'kitchen_basic':     {'include': ['kitchen', 'cooking basics', 'dishes and silverware']},
    'tv':                {'include': ['tv', 'hdtv']},
    'iron':              {'include': ['iron']},
    'hair_dryer':        {'include': ['hair dryer']},
    'toiletries':        {'include': ['shampoo', 'conditioner', 'body soap', 'shower gel']},
    'essentials':        {'include': ['essentials', 'bed linens', 'hangers', 'extra pillows']},

    # Safety baseline
    'safety_baseline':   {'include': ['smoke alarm', 'carbon monoxide', 'fire extinguisher', 'first aid']},

    # Equipped kitchen
    'dishwasher':        {'include': ['dishwasher']},
    'oven':               {'include': ['oven']},
    'microwave':         {'include': ['microwave']},
    'coffee':            {'include': ['coffee']},
    'wine_glasses':      {'include': ['wine glasses']},

    # Comfort+ / climate
    'air_conditioning':  {'include': ['air conditioning', 'ac -']},
    'washer':            {'include': ['washer']},
    'dryer':             {'include': ['dryer'], 'exclude': ['hair dryer', 'drying rack']},
    'workspace':         {'include': ['workspace', 'desk']},

    # Parking / access
    'parking_free':      {'include': ['free parking', 'free street parking', 'free residential garage']},
    'parking_paid':      {'include': ['paid parking']},
    'ev_charger':        {'include': ['ev charger']},
    'elevator':          {'include': ['elevator']},

    # Premium
    'pool':              {'include': ['pool'], 'exclude': ['pool table', 'pool view']},
    'hot_tub':           {'include': ['hot tub']},
    'gym':               {'include': ['gym', 'exercise equipment']},
    'fireplace':         {'include': ['fireplace']},
    'bbq':               {'include': ['bbq', 'barbecue']},
    'outdoor_space':     {'include': ['backyard', 'patio', 'balcony', 'outdoor furniture',
                                      'outdoor dining', 'sun lounger', 'hammock', 'fire pit', 'garden'],
                          'exclude': ['garden view']},
    'nice_view':         {'include': ['skyline view', 'garden view', 'pool view',
                                      'lake access', 'waterfront', 'beach']},

    # Luxury / lifestyle
    'security_premium':  {'include': ['security camera', 'smart lock', 'keypad', 'lockbox']},
    'housekeeping':      {'include': ['housekeeping', 'cleaning available during stay']},
    'concierge_like':    {'include': ['building staff', 'host greets you', 'doorman', 'concierge']},
    'entertainment':     {'include': ['board games', 'sound system', 'game console',
                                      'piano', 'ping pong', 'record player', 'books and reading']},

    # Family
    'kids_friendly':     {'include': ['crib', 'high chair', 'children', 'pack ', 'playground']},

    # Permissive
    'pets_allowed':      {'include': ['pets allowed']},
    'long_term_ok':      {'include': ['long term stays allowed']},
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _token_matches_concept(token: str, concept_spec: dict) -> bool:
    """Return True if a single token matches a concept's include/exclude rules.

    Matching is case-insensitive substring matching. A token matches if it
    contains any include pattern AND none of the exclude patterns.
    """
    token_lower = token.lower()
    includes = concept_spec.get('include', [])
    excludes = concept_spec.get('exclude', [])

    has_include = any(pat in token_lower for pat in includes)
    has_exclude = any(pat in token_lower for pat in excludes)

    return has_include and not has_exclude


def _amenities_to_list(value):
    """Parse a single amenities cell into a Python list.

    Idempotent: accepts either a Python list (returned as-is) or a string
    representation of a list as produced by Inside Airbnb (parsed via
    ast.literal_eval). This lets the same module work seamlessly at fit time
    (parquet column stored as strings) and at inference time in the Streamlit
    app (real Python list from the UI).
    """
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)


def _compute_concept_presence(amenities_list: list, concept_dict: dict) -> dict:
    """For a single listing, return {concept_name: 0/1} based on its tokens."""
    return {
        concept: int(any(_token_matches_concept(tok, spec) for tok in amenities_list))
        for concept, spec in concept_dict.items()
    }


def _score_to_tier(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Vectorized mapping from continuous score to {basic, premium, luxury}."""
    return np.where(
        scores < thresholds[0],
        'basic',
        np.where(scores < thresholds[1], 'premium', 'luxury'),
    )


# ---------------------------------------------------------------------------
# Public: build the (n_listings x n_concepts) binary matrix
# ---------------------------------------------------------------------------

def build_concept_matrix(amenities: pd.Series, concept_dict: dict = CONCEPTS) -> pd.DataFrame:
    """Build a binary (n_listings x n_concepts) DataFrame of concept presence.

    Parameters
    ----------
    amenities : pd.Series
        Each value is either a list of amenity strings or its string repr
        (e.g. '["Wifi", "Heating"]').
    concept_dict : dict
        Concept dictionary. Defaults to the module-level CONCEPTS.

    Returns
    -------
    pd.DataFrame
        Index matches `amenities.index`. Columns are concept names.
        Values are 0/1 (int).
    """
    parsed = amenities.apply(_amenities_to_list)
    records = [_compute_concept_presence(tokens, concept_dict) for tokens in parsed]
    return pd.DataFrame(records, index=amenities.index)


# ---------------------------------------------------------------------------
# Public: fit the amenity_tier pipeline on a city's training data
# ---------------------------------------------------------------------------

def fit_amenity_tier(
    amenities: pd.Series,
    price: pd.Series,
    concept_dict: dict = CONCEPTS,
    prevalence_threshold: float = 0.80,
    alpha_grid: np.ndarray = None,
    n_folds: int = 5,
    random_state: int = 42,
):
    """Fit the amenity_tier pipeline on a city's training data.

    Pipeline:
      1. Parse amenities into a binary concept matrix.
      2. Drop concepts whose prevalence is >= `prevalence_threshold`.
         These near-constant baselines carry no variance and confuse Ridge.
      3. Nested-CV Ridge: outer KFold for OOF predictions, inner RidgeCV
         to pick alpha at each fold. The most frequent alpha is then used
         to refit one Ridge on the full data, which is the model returned
         for inference.
      4. Bucket the OOF predictions into 3 tiers via empirical terciles.

    Parameters
    ----------
    amenities : pd.Series
        Amenity list (or its string repr) per listing.
    price : pd.Series
        Raw price in $ per listing (NOT log-transformed). Same index as `amenities`.
    concept_dict : dict
    prevalence_threshold : float in (0, 1)
        Concepts with prevalence >= this value are dropped before Ridge.
    alpha_grid : array-like, optional
        Ridge alphas to grid-search. Defaults to np.logspace(-2, 3, 20).
    n_folds : int
    random_state : int

    Returns
    -------
    tier : pd.Series
        Ordered Categorical {basic < premium < luxury}, indexed like `amenities`.
    oof_score : np.ndarray
        Continuous OOF amenity score (in log1p space).
    bundle : dict
        Serializable bundle for `transform_amenity_tier`. See module docstring.
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-2, 3, 20)

    # Step 1: full concept matrix
    concepts_df = build_concept_matrix(amenities, concept_dict)

    # Step 2: drop high-prevalence concepts
    prevalence = concepts_df.mean()
    discriminant_concepts = prevalence[prevalence < prevalence_threshold].index.tolist()
    X = concepts_df[discriminant_concepts].values

    # Step 3: nested-CV Ridge for OOF score
    y_log = np.log1p(price.values)
    outer_cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    oof_score = cross_val_predict(
        RidgeCV(alphas=alpha_grid, cv=5),
        X,
        y_log,
        cv=outer_cv,
        n_jobs=-1,
    )

    # Identify the modal alpha across outer folds
    alphas_chosen = []
    for train_idx, _ in outer_cv.split(X):
        fold_model = RidgeCV(alphas=alpha_grid, cv=5)
        fold_model.fit(X[train_idx], y_log[train_idx])
        alphas_chosen.append(fold_model.alpha_)
    final_alpha = Counter(alphas_chosen).most_common(1)[0][0]

    # Refit one Ridge on the full data with the chosen alpha
    final_model = Ridge(alpha=final_alpha)
    final_model.fit(X, y_log)

    # Step 4: tercile-based tiering on the OOF score
    tercile_thresholds = np.quantile(oof_score, [1 / 3, 2 / 3])
    tier_values = _score_to_tier(oof_score, tercile_thresholds)
    tier = pd.Series(
        pd.Categorical(tier_values, categories=['basic', 'premium', 'luxury'], ordered=True),
        index=amenities.index,
        name='amenity_tier',
    )

    bundle = {
        'model': final_model,
        'discriminant_concepts': discriminant_concepts,
        'tercile_thresholds': tercile_thresholds,
        'alpha': float(final_alpha),
        'prevalence_threshold': prevalence_threshold,
    }

    return tier, oof_score, bundle


# ---------------------------------------------------------------------------
# Public: apply a fitted bundle to new data (inference-time)
# ---------------------------------------------------------------------------

def transform_amenity_tier(
    amenities: pd.Series,
    bundle: dict,
    concept_dict: dict = CONCEPTS,
):
    """Apply a fitted bundle to score new listings.

    Used by the Streamlit app to turn a user's amenity selection into an
    `amenity_tier` without retraining anything.

    Parameters
    ----------
    amenities : pd.Series
    bundle : dict
        As returned by `fit_amenity_tier`.
    concept_dict : dict

    Returns
    -------
    tier : pd.Series
        Ordered Categorical {basic < premium < luxury}, indexed like `amenities`.
    score : np.ndarray
        Continuous amenity score (in log1p space).
    """
    concepts_df = build_concept_matrix(amenities, concept_dict)
    X = concepts_df[bundle['discriminant_concepts']].values

    score = bundle['model'].predict(X)
    tier_values = _score_to_tier(score, bundle['tercile_thresholds'])

    tier = pd.Series(
        pd.Categorical(tier_values, categories=['basic', 'premium', 'luxury'], ordered=True),
        index=amenities.index,
        name='amenity_tier',
    )

    return tier, score