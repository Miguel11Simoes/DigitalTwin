"""
edge_runtime
============
Runtime de inferência para edge deployment.

Módulos:
- inference_pump: Motor de inferência principal
- alarm_logic: Sistema de alarmes industriais

Uso:
    from src.edge_runtime import PumpInference, AlarmManager
    
    engine = PumpInference("models/")
    alarm_mgr = AlarmManager()
    
    result = engine.predict(X_spec, X_aux, mask)
    alarms = alarm_mgr.evaluate({...})
"""

from .inference_pump import PumpInference, PredictionResult, AlarmConfig as InferenceAlarmConfig
from .alarm_logic import AlarmManager, AlarmConfig, AlarmState, AlarmLevel, AlarmType

__all__ = [
    "PumpInference",
    "PredictionResult", 
    "InferenceAlarmConfig",
    "AlarmManager",
    "AlarmConfig",
    "AlarmState",
    "AlarmLevel",
    "AlarmType",
]
