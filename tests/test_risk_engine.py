import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.risk_engine import RiskEngine


def test_risk_engine_calculations():
    engine = RiskEngine()

    # 1. Low risk scenario (genuine voice, low probability)
    low_res = engine.compute_risk(classifier_spoof_prob=0.08)
    assert low_res.risk_level == "LOW"
    assert low_res.alert_triggered is False
    assert low_res.safety_override_triggered is False

    # 2. Critical risk scenario (high spoof probability)
    crit_res = engine.compute_risk(classifier_spoof_prob=0.98)
    assert crit_res.risk_level == "CRITICAL"
    assert crit_res.alert_triggered is True

    # 3. Safety Override rule: Extreme spoof probability (0.95) with trusted caller discount
    override_res = engine.compute_risk(
        classifier_spoof_prob=0.95,
        context_metadata={"is_trusted_contact": True}
    )
    assert override_res.safety_override_triggered is True
    assert override_res.risk_score >= 85.0
    print("[+] Test passed: Multi-signal risk engine & safety override functioning as expected!")


if __name__ == "__main__":
    test_risk_engine_calculations()
