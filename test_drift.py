from app.drift import DriftDetector

def test_drift_increases_on_different_texts():
    d = DriftDetector()
    d.start_session("s1", "I want to book a flight to NYC")
    sim1, drift1, smooth1, alert1 = d.compute("s1", "I want to book a flight to NYC")
    sim2, drift2, smooth2, alert2 = d.compute("s1", "Actually, I crave pizza tonight")
    assert drift2 >= drift1
