"""
Adaptive per-address weighting for the 3-model ensemble.

The fixed ensemble hands the Isolation Forest a flat 5% weight on every
address. On most addresses that just adds noise, but on a handful of real
exploits the IF is the only model that reacts (XGB near 0, RF low, IF near
1.0). So rather than a flat weight, this module nudges the IF's share up or
down per address using a few smooth rules.

Two things drive the adjustment: how anomalous the IF thinks the address is,
and how confident the supervised pair (XGB + RF) already are. Push IF up when
it's loud and the supervised pair is quiet; back it off when the supervised
pair already has an opinion.

The rules are gated with sigmoids instead of hard if/else so the weight stays
continuous. It's effectively a tiny Mamdani-style fuzzy controller, hand-rolled
rather than pulling in scikit-fuzzy for three rules.
"""
import threading
import numpy as np


class ScoringConfig:
    """Thread-safe singleton holding the active scoring mode.

    The mode is switchable at runtime through the API, no restart needed.
    'fixed' and 'adaptive' read the same cached XGB/RF/IF scores; only the
    way they're combined differs.
    """
    _instance = None
    _lock = threading.Lock()

    VALID_MODES = ("fixed", "adaptive")

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._mode = "fixed"  # default keeps the old behaviour
        return cls._instance

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{value}'. Must be one of {self.VALID_MODES}")
        with self._lock:
            self._mode = value


scoring_config = ScoringConfig()


def _sigmoid(x: float, center: float, steepness: float) -> float:
    """Logistic membership curve in [0, 1]. center is where it crosses 0.5,
    steepness controls how sharp the transition is."""
    z = steepness * (x - center)
    z = max(-500.0, min(500.0, z))  # keep exp from overflowing
    return 1.0 / (1.0 + np.exp(-z))


# Base supervised split. These come from the validation inverse-error
# weighting and stay fixed regardless of which IF model is loaded.
_W_XGB_BASE = 0.190
_W_RF_BASE = 0.810

# Bounds on the IF share. Never let it drop to zero or run away with the score.
_W_IF_MIN = 0.02
_W_IF_MAX = 0.30

# Two tuned rule sets. V1 was fitted against the early (inverse-correlation)
# IF; V2 was re-fitted against the Phase 3 rank-calibrated IF via the grid
# search in scripts/grid_search_adaptive_rules.py. Same rule structure either
# way - only the sigmoid centres, steepnesses and gains move. install_rule_params
# swaps the active set at startup.
RULE_PARAMS_V1: dict = {
    "W_IF_MIN": 0.02,
    "W_IF_MAX": 0.30,
    "R1_IF_CENTER": 0.6,
    "R1_IF_STEEPNESS": 15.0,
    "R1_SUP_CENTER": 0.15,
    "R1_SUP_STEEPNESS": 20.0,
    "R1_GAIN": 0.25,
    "R2_SUP_CENTER": 0.15,
    "R2_SUP_STEEPNESS": 20.0,
    "R2_GAIN": 0.03,
    "R3_AGREE_CENTER": 0.80,
    "R3_AGREE_STEEPNESS": 15.0,
    "R3_GAIN": 0.05,
    "BASE_W_IF": 0.05,
}

# Live copies the rule functions actually read. install_rule_params overwrites
# these; they start on V1.
_R1_IF_CENTER = RULE_PARAMS_V1["R1_IF_CENTER"]
_R1_IF_STEEPNESS = RULE_PARAMS_V1["R1_IF_STEEPNESS"]
_R1_SUP_CENTER = RULE_PARAMS_V1["R1_SUP_CENTER"]
_R1_SUP_STEEPNESS = RULE_PARAMS_V1["R1_SUP_STEEPNESS"]
_R1_GAIN = RULE_PARAMS_V1["R1_GAIN"]
_R2_SUP_CENTER = RULE_PARAMS_V1["R2_SUP_CENTER"]
_R2_SUP_STEEPNESS = RULE_PARAMS_V1["R2_SUP_STEEPNESS"]
_R2_GAIN = RULE_PARAMS_V1["R2_GAIN"]
_R3_AGREE_CENTER = RULE_PARAMS_V1["R3_AGREE_CENTER"]
_R3_AGREE_STEEPNESS = RULE_PARAMS_V1["R3_AGREE_STEEPNESS"]
_R3_GAIN = RULE_PARAMS_V1["R3_GAIN"]
_BASE_W_IF = RULE_PARAMS_V1["BASE_W_IF"]
_ACTIVE_VERSION: str = "v1"


def install_rule_params(params: dict, version: str = "custom") -> None:
    """Swap in a different rule-parameter set at runtime.

    params must carry exactly the keys in RULE_PARAMS_V1 (mismatches raise).
    version is just a label for logging. No locking here - this only runs once
    at detector startup before any scoring threads exist.
    """
    global _R1_IF_CENTER, _R1_IF_STEEPNESS
    global _R1_SUP_CENTER, _R1_SUP_STEEPNESS, _R1_GAIN
    global _R2_SUP_CENTER, _R2_SUP_STEEPNESS, _R2_GAIN
    global _R3_AGREE_CENTER, _R3_AGREE_STEEPNESS, _R3_GAIN
    global _W_IF_MIN, _W_IF_MAX, _BASE_W_IF, _ACTIVE_VERSION

    required = set(RULE_PARAMS_V1.keys())
    got = set(params.keys())
    if got != required:
        missing = required - got
        extra = got - required
        raise ValueError(f"rule-params mismatch — missing {missing}, extra {extra}")

    _W_IF_MIN = float(params["W_IF_MIN"])
    _W_IF_MAX = float(params["W_IF_MAX"])
    _R1_IF_CENTER = float(params["R1_IF_CENTER"])
    _R1_IF_STEEPNESS = float(params["R1_IF_STEEPNESS"])
    _R1_SUP_CENTER = float(params["R1_SUP_CENTER"])
    _R1_SUP_STEEPNESS = float(params["R1_SUP_STEEPNESS"])
    _R1_GAIN = float(params["R1_GAIN"])
    _R2_SUP_CENTER = float(params["R2_SUP_CENTER"])
    _R2_SUP_STEEPNESS = float(params["R2_SUP_STEEPNESS"])
    _R2_GAIN = float(params["R2_GAIN"])
    _R3_AGREE_CENTER = float(params["R3_AGREE_CENTER"])
    _R3_AGREE_STEEPNESS = float(params["R3_AGREE_STEEPNESS"])
    _R3_GAIN = float(params["R3_GAIN"])
    _BASE_W_IF = float(params["BASE_W_IF"])
    _ACTIVE_VERSION = str(version)


def get_active_rule_params() -> dict:
    """Snapshot of the currently active rule parameters."""
    return {
        "version": _ACTIVE_VERSION,
        "W_IF_MIN": _W_IF_MIN,
        "W_IF_MAX": _W_IF_MAX,
        "R1_IF_CENTER": _R1_IF_CENTER,
        "R1_IF_STEEPNESS": _R1_IF_STEEPNESS,
        "R1_SUP_CENTER": _R1_SUP_CENTER,
        "R1_SUP_STEEPNESS": _R1_SUP_STEEPNESS,
        "R1_GAIN": _R1_GAIN,
        "R2_SUP_CENTER": _R2_SUP_CENTER,
        "R2_SUP_STEEPNESS": _R2_SUP_STEEPNESS,
        "R2_GAIN": _R2_GAIN,
        "R3_AGREE_CENTER": _R3_AGREE_CENTER,
        "R3_AGREE_STEEPNESS": _R3_AGREE_STEEPNESS,
        "R3_GAIN": _R3_GAIN,
        "BASE_W_IF": _BASE_W_IF,
    }


def compute_adaptive_weights(
    xgb_prob: float,
    rf_prob: float,
    if_score: float,
) -> tuple[float, float, float]:
    """Work out the per-address (w_xgb, w_rf, w_if) split. Inputs are the three
    model scores in [0, 1]; the returned weights sum to 1.

    Three rules adjust the IF share around its 0.05 base:
      R1  amplify  - IF loud + supervised quiet. The exploit case (Ronin,
                     Euler, etc.) where XGB~0 and RF<0.2 but IF>0.9.
      R2  dampen   - IF loud but supervised already has some confidence. Stops
                     IF blowing up false positives on the likes of the Uniswap
                     V2 router (supervised ~0.26, IF ~1.0).
      R3  tiebreak - IF loud + XGB and RF disagree. Small nudge for boundary
                     cases.
    When the IF is quiet none of them fire and the base weight stands.
    """
    supervised_avg = _W_XGB_BASE * xgb_prob + _W_RF_BASE * rf_prob
    agreement = 1.0 - abs(xgb_prob - rf_prob)

    # membership degrees for each antecedent
    mu_high_anomaly = _sigmoid(if_score, _R1_IF_CENTER, _R1_IF_STEEPNESS)
    mu_low_supervised = 1.0 - _sigmoid(supervised_avg, _R1_SUP_CENTER, _R1_SUP_STEEPNESS)
    mu_not_low_supervised = _sigmoid(supervised_avg, _R2_SUP_CENTER, _R2_SUP_STEEPNESS)
    mu_low_agreement = 1.0 - _sigmoid(agreement, _R3_AGREE_CENTER, _R3_AGREE_STEEPNESS)

    r1_amplify = mu_high_anomaly * mu_low_supervised * _R1_GAIN
    r2_dampen = mu_high_anomaly * mu_not_low_supervised * _R2_GAIN
    r3_tiebreak = mu_high_anomaly * mu_low_agreement * _R3_GAIN

    w_if = _BASE_W_IF + r1_amplify - r2_dampen + r3_tiebreak
    w_if = float(np.clip(w_if, _W_IF_MIN, _W_IF_MAX))

    # give the rest to XGB and RF, keeping their base proportions
    w_supervised = 1.0 - w_if
    ratio_sum = _W_XGB_BASE + _W_RF_BASE  # = 1.0
    w_xgb = w_supervised * (_W_XGB_BASE / ratio_sum)
    w_rf = w_supervised * (_W_RF_BASE / ratio_sum)

    return (round(w_xgb, 6), round(w_rf, 6), round(w_if, 6))


def adaptive_ensemble_score(
    xgb_prob: float,
    rf_prob: float,
    if_score: float,
) -> tuple[float, tuple[float, float, float]]:
    """Final adaptive probability plus the weights used. Returns
    (final_prob, (w_xgb, w_rf, w_if))."""
    w_xgb, w_rf, w_if = compute_adaptive_weights(xgb_prob, rf_prob, if_score)
    final_prob = w_xgb * xgb_prob + w_rf * rf_prob + w_if * if_score
    return (round(float(final_prob), 6), (w_xgb, w_rf, w_if))


# Array versions of the two functions above, used when scoring the whole
# holdout at once. Same maths, just vectorised over numpy arrays.

def compute_adaptive_weights_batch(
    xgb_probs: np.ndarray,
    rf_probs: np.ndarray,
    if_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compute_adaptive_weights over arrays. Returns three (n,) weight arrays."""
    supervised_avg = _W_XGB_BASE * xgb_probs + _W_RF_BASE * rf_probs
    agreement = 1.0 - np.abs(xgb_probs - rf_probs)

    def sig(x, c, s):
        z = np.clip(s * (x - c), -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    mu_high_anomaly = sig(if_scores, _R1_IF_CENTER, _R1_IF_STEEPNESS)
    mu_low_supervised = 1.0 - sig(supervised_avg, _R1_SUP_CENTER, _R1_SUP_STEEPNESS)
    mu_not_low_supervised = sig(supervised_avg, _R2_SUP_CENTER, _R2_SUP_STEEPNESS)
    mu_low_agreement = 1.0 - sig(agreement, _R3_AGREE_CENTER, _R3_AGREE_STEEPNESS)

    r1 = mu_high_anomaly * mu_low_supervised * _R1_GAIN
    r2 = mu_high_anomaly * mu_not_low_supervised * _R2_GAIN
    r3 = mu_high_anomaly * mu_low_agreement * _R3_GAIN

    w_if = np.clip(_BASE_W_IF + r1 - r2 + r3, _W_IF_MIN, _W_IF_MAX)
    w_supervised = 1.0 - w_if
    w_xgb = w_supervised * (_W_XGB_BASE / (_W_XGB_BASE + _W_RF_BASE))
    w_rf = w_supervised * (_W_RF_BASE / (_W_XGB_BASE + _W_RF_BASE))

    return w_xgb, w_rf, w_if


def adaptive_ensemble_score_batch(
    xgb_probs: np.ndarray,
    rf_probs: np.ndarray,
    if_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """adaptive_ensemble_score over arrays.
    Returns (final_probs, w_xgb_arr, w_rf_arr, w_if_arr)."""
    w_xgb, w_rf, w_if = compute_adaptive_weights_batch(xgb_probs, rf_probs, if_scores)
    final = w_xgb * xgb_probs + w_rf * rf_probs + w_if * if_scores
    return final, w_xgb, w_rf, w_if
