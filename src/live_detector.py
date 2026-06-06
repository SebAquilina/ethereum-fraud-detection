"""
The live scorer. Loads the trained models once and scores an Ethereum
address on demand: pull its history off Etherscan, build the feature vector,
run the three models, combine them.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# let the sibling modules import without a package install
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from etherscan_client import get_address_data
from feature_engine import compute_features
from adaptive_ensemble import (
    scoring_config, adaptive_ensemble_score,
    install_rule_params, get_active_rule_params,
)
from if_preprocessing import apply_recipe_to_vector, apply_rank_calibrator


class FraudDetector:
    """Scores a single Ethereum address against the trained ensemble."""

    # Default supervised split. These come from inverse-error weighting on the
    # validation AUCs: w_i proportional to 1/(1-AUC). XGB validated at ~0.992
    # and RF at ~0.998, so RF ends up carrying most of the weight. The IF gets
    # a small fixed share. Retrain the models and these need recomputing.
    # (When USE_CHRONSPLIT_MODELS is on these get overridden below.)
    _W_XGB = 0.190
    _W_RF  = 0.810
    _W_IF  = 0.05

    def __init__(self, models_dir: str = None):
        """Load everything up front. models_dir defaults to the bundled
        models/baseline_models folder."""
        if models_dir is None:
            models_dir = os.path.join(SCRIPT_DIR, "..", "models", "baseline_models")

        self.models_dir = models_dir
        self.models_loaded = False

        self._load_models()
        self._load_feature_names()

    def _load_models(self):
        """Load XGB, RF, the IF (+ its calibration) and the XGB calibrator."""
        print("Loading fraud detection models...")

        try:
            # XGB and RF default to the original-split models. Setting
            # USE_CHRONSPLIT_MODELS=1 swaps in the chronological-split retrains
            # (and their matching XGB calibrator further down).
            use_chronsplit = os.getenv("USE_CHRONSPLIT_MODELS", "0") == "1"
            self.supervised_variant = "chronsplit" if use_chronsplit else "buggy_split"

            # Under chronsplit the validation set is tiny on the fraud side, so
            # both supervised models hit AUC=1.0 and the inverse-error weighting
            # degenerates to an even 50/50. IF stays at 0.05 (the weight sweep
            # was flat there too).
            if use_chronsplit:
                self._W_XGB = 0.50
                self._W_RF = 0.50
                self._W_IF = 0.05
                print(f"  ✓ Ensemble weights overridden for chronsplit: "
                      f"w_xgb={self._W_XGB}, w_rf={self._W_RF}, w_if={self._W_IF}")

            xgb_path_default = os.path.join(self.models_dir, "xgb_baseline.json")
            xgb_path_chronsplit = os.path.join(self.models_dir, "xgboost_chronsplit.json")
            xgb_path = xgb_path_chronsplit if (use_chronsplit and os.path.exists(xgb_path_chronsplit)) else xgb_path_default
            if os.path.exists(xgb_path):
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(xgb_path)
                print(f"  ✓ XGBoost loaded ({'chronsplit' if xgb_path == xgb_path_chronsplit else 'buggy-split baseline'})")
            else:
                print(f"  ✗ XGBoost not found at {xgb_path}")
                self.xgb_model = None

            rf_path_default = os.path.join(self.models_dir, "rf_baseline.pkl")
            rf_path_chronsplit = os.path.join(self.models_dir, "random_forest_chronsplit.pkl")
            rf_path = rf_path_chronsplit if (use_chronsplit and os.path.exists(rf_path_chronsplit)) else rf_path_default
            if os.path.exists(rf_path):
                self.rf_model = joblib.load(rf_path)
                print(f"  ✓ Random Forest loaded ({'chronsplit' if rf_path == rf_path_chronsplit else 'buggy-split baseline'})")
            else:
                print(f"  ✗ Random Forest not found at {rf_path}")
                self.rf_model = None
            
            # The IF went through a few iterations and they're all still here so
            # results stay reproducible. Default is the Phase 3 model (with the
            # log/drop recipe + rank calibration); the older ones are reachable
            # via env vars:
            #   USE_PHASE2_IF=1  -> Phase 2 (recipe + percentile calibration)
            #   USE_PHASE1_IF=1  -> Phase 1 (legit-only, no recipe)
            #   USE_LEGACY_IF=1  -> the original mixed-data model
            use_legacy_if = os.getenv("USE_LEGACY_IF", "0") == "1"
            use_phase1_if = os.getenv("USE_PHASE1_IF", "0") == "1"
            use_phase2_if = os.getenv("USE_PHASE2_IF", "0") == "1"

            phase3_if_path = os.path.join(self.models_dir, "isolation_forest_phase3.pkl")
            phase3_params_path = os.path.join(self.models_dir, "if_score_params_phase3.joblib")
            phase2_if_path = os.path.join(self.models_dir, "isolation_forest_phase2.pkl")
            phase2_params_path = os.path.join(self.models_dir, "if_score_params_phase2.joblib")
            clean_if_path = os.path.join(self.models_dir, "isolation_forest_clean.pkl")
            clean_params_path = os.path.join(self.models_dir, "if_score_params_clean.joblib")
            legacy_if_path = os.path.join(self.models_dir, "unsupervised_static_if.pkl")
            legacy_params_path = os.path.join(self.models_dir, "if_score_params.joblib")
            v2_rules_path = os.path.join(self.models_dir, "adaptive_rule_params_v2.joblib")

            def _load_recipe(params):
                return params.get("recipe") if params else None

            def _load_calibrator(params):
                return params.get("rank_calibrator") if params else None

            if use_legacy_if and os.path.exists(legacy_if_path):
                self.if_model = joblib.load(legacy_if_path)
                self.if_variant = "legacy"
                self.if_recipe = None
                self.if_calibrator = None
                self.if_params = joblib.load(legacy_params_path) if os.path.exists(legacy_params_path) else None
                print("  ✓ Isolation Forest loaded (legacy) — USE_LEGACY_IF=1")
            elif use_phase1_if and os.path.exists(clean_if_path):
                self.if_model = joblib.load(clean_if_path)
                self.if_variant = "clean"
                self.if_recipe = None
                self.if_calibrator = None
                self.if_params = joblib.load(clean_params_path) if os.path.exists(clean_params_path) else None
                print("  ✓ Isolation Forest loaded (Phase 1 clean) — USE_PHASE1_IF=1")
            elif use_phase2_if and os.path.exists(phase2_if_path):
                self.if_model = joblib.load(phase2_if_path)
                self.if_variant = "phase2"
                self.if_params = joblib.load(phase2_params_path) if os.path.exists(phase2_params_path) else None
                self.if_recipe = _load_recipe(self.if_params)
                self.if_calibrator = None
                print("  ✓ Isolation Forest loaded (Phase 2 — log+drop + percentile) — USE_PHASE2_IF=1")
            elif os.path.exists(phase3_if_path):
                self.if_model = joblib.load(phase3_if_path)
                self.if_variant = "phase3"
                # Same IF trees and recipe either way; this just swaps in the
                # calibrator ECDF that was refit on the corrected legit set.
                use_chronsplit_if_calib = os.getenv("USE_CHRONSPLIT_IF_CALIB", "0") == "1"
                chronsplit_calib_path = os.path.join(self.models_dir, "if_score_params_chronsplit.joblib")
                if use_chronsplit_if_calib and os.path.exists(chronsplit_calib_path):
                    self.if_params = joblib.load(chronsplit_calib_path)
                    self.if_variant = "phase3+chronsplit_calib"
                    print("  ✓ Isolation Forest loaded (Phase 3 model) + chronsplit rank calibrator — USE_CHRONSPLIT_IF_CALIB=1")
                else:
                    self.if_params = joblib.load(phase3_params_path) if os.path.exists(phase3_params_path) else None
                    print("  ✓ Isolation Forest loaded (Phase 3 — rank calibration + log+drop)")
                self.if_recipe = _load_recipe(self.if_params)
                self.if_calibrator = _load_calibrator(self.if_params)
                if self.if_recipe is not None:
                    print(f"    recipe: {self.if_recipe['n_log']} log-transformed, "
                          f"{self.if_recipe['n_drop']} dropped, {self.if_recipe['n_kept']} kept")
                if self.if_calibrator is not None:
                    print(f"    rank calibrator: n_training={self.if_calibrator['n_training']}")
            elif os.path.exists(phase2_if_path):
                self.if_model = joblib.load(phase2_if_path)
                self.if_variant = "phase2"
                self.if_params = joblib.load(phase2_params_path) if os.path.exists(phase2_params_path) else None
                self.if_recipe = _load_recipe(self.if_params)
                self.if_calibrator = None
                print("  ✓ Isolation Forest loaded (Phase 2) — Phase 3 artefact missing")
            elif os.path.exists(clean_if_path):
                self.if_model = joblib.load(clean_if_path)
                self.if_variant = "clean"
                self.if_recipe = None
                self.if_calibrator = None
                self.if_params = joblib.load(clean_params_path) if os.path.exists(clean_params_path) else None
                print("  ✓ Isolation Forest loaded (Phase 1 clean)")
            elif os.path.exists(legacy_if_path):
                self.if_model = joblib.load(legacy_if_path)
                self.if_variant = "legacy"
                self.if_recipe = None
                self.if_calibrator = None
                self.if_params = joblib.load(legacy_params_path) if os.path.exists(legacy_params_path) else None
                print("  ✓ Isolation Forest loaded (legacy)")
            else:
                print("  ✗ No Isolation Forest model found")
                self.if_model = None
                self.if_variant = None
                self.if_recipe = None
                self.if_calibrator = None
                self.if_params = None

            # V2 adaptive rules go with the Phase 3 IF. If an older IF was
            # forced via env var, leave the rules on V1.
            if self.if_variant == "phase3" and os.path.exists(v2_rules_path):
                try:
                    v2_blob = joblib.load(v2_rules_path)
                    install_rule_params(v2_blob["params"], version=v2_blob.get("version", "v2"))
                    print(f"  ✓ Adaptive rule parameters installed ({get_active_rule_params()['version']}): "
                          f"R1_IF_CENTER={v2_blob['params']['R1_IF_CENTER']}, "
                          f"R1_GAIN={v2_blob['params']['R1_GAIN']}, "
                          f"R2_GAIN={v2_blob['params']['R2_GAIN']}")
                except Exception as e:
                    print(f"  ! Failed to install V2 rule params: {e} — keeping V1")
            else:
                print(f"  ○ Adaptive rule parameters: v1 (IF variant={self.if_variant})")

            # The LightGBM stacker (Tier 1) is deliberately off. It overfit
            # badly on live data - basically no recall on the known exploits and
            # wild false positives on multi-sig wallets (Gnosis Safe came out at
            # 0.97). The weighted average is what actually runs; the stacker is
            # only kept around for the write-up.
            self.stacker = None
            print("  ○ Stacker disabled — Tier 2 (weighted average) active")

            # XGB calibrator. The chronsplit one is paired with the chronsplit
            # XGB (both retrained on the corrected split).
            iso_path_default = os.path.join(self.models_dir, "iso_calibrator.joblib")
            iso_path_chronsplit = os.path.join(self.models_dir, "xgboost_chronsplit_calibrated.pkl")
            iso_path = iso_path_chronsplit if (use_chronsplit and os.path.exists(iso_path_chronsplit)) else iso_path_default
            if os.path.exists(iso_path):
                self.calibrator = joblib.load(iso_path)
                print(f"  ✓ Isotonic calibrator loaded ({'chronsplit' if iso_path == iso_path_chronsplit else 'buggy-split'})")
            else:
                print("  ○ Calibrator not found - using raw probabilities")
                self.calibrator = None
            
            self.models_loaded = self.xgb_model is not None or self.rf_model is not None
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models_loaded = False
    
    def _load_feature_names(self):
        """Read the column order straight off the training CSV header. Falls
        back to a hardcoded copy if the CSV isn't shipped."""
        train_path = os.path.join(SCRIPT_DIR, "..", "data", "processed", "prepared_data_1", "train.csv")
        if os.path.exists(train_path):
            df = pd.read_csv(train_path, nrows=1)
            self.feature_names = [c for c in df.columns if c not in ['Address', 'FLAG',
                                                                       'ERC20 most sent token type',
                                                                       'ERC20_most_rec_token_type']]
            print(f"  ✓ Loaded {len(self.feature_names)} feature names")
        else:
            self.feature_names = self._get_default_feature_names()
            print(f"  ○ Using default {len(self.feature_names)} feature names")

    def _get_default_feature_names(self):
        """The training column order, hardcoded for when the CSV isn't around."""
        return [
            'Avg min between sent tnx', 'Avg min between received tnx',
            'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
            'Number of Created Contracts', 'Unique Received From Addresses',
            'Unique Sent To Addresses', 'min value received', 'max value received',
            'avg val received', 'min val sent', 'max val sent', 'avg val sent',
            'min value sent to contract', 'max val sent to contract',
            'avg value sent to contract', 'total transactions (including tnx to create contract',
            'total Ether sent', 'total ether received', 'total ether sent contracts',
            'total ether balance', 'Total ERC20 tnxs', 'ERC20 total Ether received',
            'ERC20 total ether sent', 'ERC20 total Ether sent contract',
            'ERC20 uniq sent addr', 'ERC20 uniq rec addr', 'ERC20 uniq sent addr.1',
            'ERC20 uniq rec contract addr', 'ERC20 avg time between sent tnx',
            'ERC20 avg time between rec tnx', 'ERC20 avg time between rec 2 tnx',
            'ERC20 avg time between contract tnx', 'ERC20 min val rec', 'ERC20 max val rec',
            'ERC20 avg val rec', 'ERC20 min val sent', 'ERC20 max val sent',
            'ERC20 avg val sent', 'ERC20 min val sent contract', 'ERC20 max val sent contract',
            'ERC20 avg val sent contract', 'ERC20 uniq sent token name',
            'ERC20 uniq rec token name', 'decile', 'log_ether_sent',
            'tx_count_last_60m', 'ether_sent_last_60m', 'tx_count_last_360m',
            'ether_sent_last_360m', 'tx_count_last_1440m', 'ether_sent_last_1440m',
            'tx_count_last_10080m', 'ether_sent_last_10080m', 'ratio_vs_avg_60m',
            'ratio_vs_avg_360m', 'ratio_vs_avg_1440m', 'ratio_vs_avg_10080m'
        ]
    
    def score_address(self, address: str) -> dict:
        """Score one address. Pass it with or without the 0x prefix. Returns a
        dict with the fraud probability, risk band and the individual model
        scores (or an error dict if there's no on-chain history)."""
        if not self.models_loaded:
            return {"error": "Models not loaded", "address": address}

        if not address.startswith("0x"):
            address = "0x" + address
        address = address.lower()

        print(f"\n{'='*60}")
        print(f"Scoring address: {address}")
        print(f"{'='*60}")

        data = get_address_data(address)

        if not data.get("normal_transactions") and not data.get("erc20_transfers"):
            return {
                "address": address,
                "error": "No transaction history found",
                "fraud_probability": None,
                "risk_level": "UNKNOWN"
            }

        features = compute_features(data)

        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
        # scrub NaN/inf and keep things inside float32 range - some whale
        # addresses produce values big enough to trip up XGBoost otherwise
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        max_val = np.finfo(np.float32).max * 0.9  # ~3.06e38
        feature_vector = np.clip(feature_vector, -max_val, max_val)

        dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=self.feature_names)
        xgb_prob = float(self.xgb_model.predict(dmatrix)[0])
        if self.calibrator:
            xgb_prob = float(self.calibrator.predict([xgb_prob])[0])

        rf_prob = None
        if self.rf_model:
            rf_prob = float(self.rf_model.predict_proba(feature_vector.reshape(1, -1))[0, 1])

        # IF score. Phase 2/3 run the recipe over the vector first; Phase 3 then
        # maps the raw anomaly through the rank calibrator, older variants use
        # the q01/q99 percentile clip.
        if_score = None
        if self.if_model:
            if self.if_recipe is not None:
                if_input = apply_recipe_to_vector(feature_vector, self.feature_names, self.if_recipe)
            else:
                if_input = feature_vector
            raw_score = self.if_model.decision_function(if_input.reshape(1, -1))[0]
            anomaly_score = -raw_score  # flip so bigger = more anomalous
            if self.if_calibrator is not None:
                if_score = apply_rank_calibrator(float(anomaly_score), self.if_calibrator)
            elif self.if_params:
                q01, q99 = self.if_params['q01'], self.if_params['q99']
                if_score = float(np.clip((anomaly_score - q01) / (q99 - q01 + 1e-8), 0, 1))
            else:
                if_score = max(0.0, min(1.0, 0.5 - raw_score / 2))

        # Pick the best combination we can given which models actually produced
        # a score, falling back down the tiers if one is missing:
        #   T1 stacker (off) -> T2 3-model -> T3 2-model -> T4 XGB only.
        # In adaptive mode T2 swaps the fixed weights for the per-address ones.
        adaptive_weights = None

        if self.stacker and rf_prob is not None and if_score is not None:
            # Tier 1, stacked meta-learner. Disabled, so this never runs.
            stack_features = np.array([[xgb_prob, rf_prob, if_score, if_score]])
            final_prob = float(self.stacker.predict_proba(stack_features)[0, 1])
            ensemble_method = "stacked_generalisation"
        elif rf_prob is not None and if_score is not None:
            active_mode = scoring_config.mode
            if active_mode == "adaptive":
                try:
                    final_prob, adaptive_weights = adaptive_ensemble_score(
                        xgb_prob, rf_prob, if_score
                    )
                    ensemble_method = "adaptive_weighted_3model"
                except Exception as e:
                    # if the adaptive maths throws for some reason, just use the
                    # fixed weights rather than failing the whole request
                    print(f"  Adaptive ensemble failed ({e}), falling back to fixed Tier 2")
                    supervised = self._W_XGB * xgb_prob + self._W_RF * rf_prob
                    final_prob = (1 - self._W_IF) * supervised + self._W_IF * if_score
                    ensemble_method = "weighted_average_3model"
                    adaptive_weights = None
            else:
                # Tier 2: supervised pair carries most of it, IF gets W_IF
                supervised = self._W_XGB * xgb_prob + self._W_RF * rf_prob
                final_prob = (1 - self._W_IF) * supervised + self._W_IF * if_score
                ensemble_method = "weighted_average_3model"
        elif rf_prob is not None:
            # Tier 3: no IF, just the two supervised models
            final_prob = self._W_XGB * xgb_prob + self._W_RF * rf_prob
            ensemble_method = "weighted_average_2model"
        else:
            # Tier 4: nothing but calibrated XGB left
            final_prob = xgb_prob
            ensemble_method = "xgboost_only"

        # bucket the probability into a risk band
        if final_prob >= 0.9:
            risk_level = "CRITICAL"
        elif final_prob >= 0.7:
            risk_level = "HIGH"
        elif final_prob >= 0.5:
            risk_level = "MEDIUM"
        elif final_prob >= 0.3:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        result = {
            "address": address,
            "fraud_probability": round(final_prob, 4),
            "risk_level": risk_level,
            "scoring_mode": scoring_config.mode,
            "ensemble_method": ensemble_method,
            "model_scores": {
                "xgboost": round(xgb_prob, 4),
                "random_forest": round(rf_prob, 4) if rf_prob is not None else None,
                "isolation_forest": round(if_score, 4) if if_score is not None else None,
            },
            "address_stats": {
                "total_transactions": len(data.get("normal_transactions", [])),
                "total_erc20_transfers": len(data.get("erc20_transfers", [])),
                "balance_eth": round(data.get("balance", 0), 4),
                "total_ether_sent": round(features.get("total Ether sent", 0), 4),
                "total_ether_received": round(features.get("total ether received", 0), 4),
            },
            "features": features,
        }

        if adaptive_weights is not None:
            result["adaptive_weights"] = {
                "w_xgb": round(adaptive_weights[0], 4),
                "w_rf": round(adaptive_weights[1], 4),
                "w_if": round(adaptive_weights[2], 4),
            }
        
        print(f"\n{'='*60}")
        print(f"FRAUD DETECTION RESULT")
        print(f"{'='*60}")
        print(f"Address:          {address}")
        print(f"Fraud Probability: {final_prob:.2%}")
        print(f"Risk Level:        {risk_level}")
        print(f"{'='*60}")
        
        return result


    def score_address_compare(self, address: str) -> dict:
        """Score once and return both the fixed and adaptive results, computed
        from the same base-model scores (one Etherscan fetch). Backs the
        "Compare modes" button in the UI."""
        # run it in fixed mode first, restoring whatever mode was set after
        original_mode = scoring_config.mode
        try:
            scoring_config.mode = "fixed"
            fixed_result = self.score_address(address)
        finally:
            scoring_config.mode = original_mode

        if fixed_result.get("error") or fixed_result.get("fraud_probability") is None:
            return {"fixed": fixed_result, "adaptive": fixed_result}

        # reuse the same base scores for the adaptive number, no second fetch
        ms = fixed_result["model_scores"]
        xgb_prob = ms["xgboost"]
        rf_prob = ms.get("random_forest")
        if_score = ms.get("isolation_forest")

        if rf_prob is not None and if_score is not None:
            ada_prob, ada_weights = adaptive_ensemble_score(xgb_prob, rf_prob, if_score)
            ada_risk = (
                "CRITICAL" if ada_prob >= 0.9 else
                "HIGH" if ada_prob >= 0.7 else
                "MEDIUM" if ada_prob >= 0.5 else
                "LOW" if ada_prob >= 0.3 else
                "MINIMAL"
            )
            adaptive_result = {
                **fixed_result,
                "fraud_probability": round(ada_prob, 4),
                "risk_level": ada_risk,
                "scoring_mode": "adaptive",
                "ensemble_method": "adaptive_weighted_3model",
                "adaptive_weights": {
                    "w_xgb": round(ada_weights[0], 4),
                    "w_rf": round(ada_weights[1], 4),
                    "w_if": round(ada_weights[2], 4),
                },
            }
        else:
            adaptive_result = fixed_result

        return {"fixed": fixed_result, "adaptive": adaptive_result}


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Score Ethereum addresses for fraud")
    parser.add_argument("address", nargs="?", help="Ethereum address to score")
    parser.add_argument("--test", action="store_true", help="Run with test address")
    args = parser.parse_args()
    
    detector = FraudDetector()
    
    if args.test:
        # Test with a known address from training data
        test_addr = "0x00009277775ac7d0d59eaad8fee3d10ac6c805e8"
        result = detector.score_address(test_addr)
    elif args.address:
        result = detector.score_address(args.address)
    else:
        print("Usage: python live_detector.py <ethereum_address>")
        print("       python live_detector.py --test")
        sys.exit(1)
    
    # Output result as JSON
    import json
    print("\nJSON Output:")
    print(json.dumps(result, indent=2))
