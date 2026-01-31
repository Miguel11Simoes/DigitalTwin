#!/usr/bin/env python3
"""
alarm_logic.py
==============
Lógica de alarmes industriais para bomba.

Implementa:
- Thresholds configuráveis
- Múltiplos níveis de alarme
- Hysteresis para evitar flapping
- Integração com sistema de notificações

Author: Industrial Pump Digital Twin
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json


class AlarmLevel(Enum):
    """Níveis de alarme."""
    NONE = 0
    INFO = 1
    WARNING = 2
    CRITICAL = 3
    EMERGENCY = 4


class AlarmType(Enum):
    """Tipos de alarme."""
    SEVERITY = "severity"
    RUL = "rul"
    HEALTH = "health"
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    OOD = "ood"
    DRIFT = "drift"
    SENSOR_FAULT = "sensor_fault"


@dataclass
class AlarmThreshold:
    """Threshold individual para um alarme."""
    
    param: str                  # Nome do parâmetro monitorado
    alarm_type: AlarmType       # Tipo de alarme
    
    # Thresholds (low e high são opcionais)
    low_critical: Optional[float] = None
    low_warning: Optional[float] = None
    high_warning: Optional[float] = None
    high_critical: Optional[float] = None
    
    # Hysteresis
    hysteresis: float = 0.05    # 5% de histerese
    
    # Tempo mínimo para alarme (evita transientes)
    min_duration_sec: float = 5.0
    
    # Ativo
    enabled: bool = True


@dataclass
class AlarmState:
    """Estado atual de um alarme."""
    
    param: str
    level: AlarmLevel
    message: str
    value: float
    threshold: float
    triggered_at: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    cleared: bool = False
    cleared_at: Optional[datetime] = None


@dataclass
class AlarmConfig:
    """Configuração completa de alarmes."""
    
    # Thresholds padrão
    severity_critical_prob: float = 0.8
    severity_degraded_prob: float = 0.6
    
    rul_critical_min: float = 60.0      # 1 hora
    rul_warning_min: float = 480.0      # 8 horas
    
    health_critical_pct: float = 30.0
    health_warning_pct: float = 50.0
    
    vibration_warning: float = 4.5      # mm/s RMS
    vibration_critical: float = 7.1     # mm/s RMS (ISO 10816-3)
    
    temperature_warning: float = 70.0    # °C
    temperature_critical: float = 85.0   # °C
    
    pressure_low_warning: float = 0.5    # bar
    pressure_low_critical: float = 0.2   # bar
    pressure_high_warning: float = 8.0   # bar
    pressure_high_critical: float = 10.0 # bar
    
    # Configurações gerais
    hysteresis_pct: float = 5.0
    min_duration_sec: float = 5.0
    max_active_alarms: int = 100
    
    # Notificações
    notify_on_warning: bool = True
    notify_on_critical: bool = True
    notify_on_clear: bool = True


class AlarmManager:
    """
    Gerenciador de alarmes industriais.
    
    Features:
    - Avaliação contínua de thresholds
    - Hysteresis para evitar flapping
    - Histórico de alarmes
    - Acknowledge e clear
    """
    
    def __init__(self, config: Optional[AlarmConfig] = None):
        """
        Inicializa gerenciador.
        
        Args:
            config: Configuração de alarmes
        """
        self.config = config or AlarmConfig()
        
        # Thresholds ativos
        self.thresholds: Dict[str, AlarmThreshold] = {}
        
        # Estados de alarme ativos
        self.active_alarms: Dict[str, AlarmState] = {}
        
        # Histórico
        self.alarm_history: List[AlarmState] = []
        
        # Tempo mínimo em alarme (para evitar transientes)
        self._pending_alarms: Dict[str, Tuple[datetime, AlarmLevel, float]] = {}
        
        # Setup thresholds padrão
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Configura thresholds padrão."""
        
        cfg = self.config
        
        # Severidade (só alto)
        self.add_threshold(AlarmThreshold(
            param="severity_critical_prob",
            alarm_type=AlarmType.SEVERITY,
            high_warning=cfg.severity_degraded_prob,
            high_critical=cfg.severity_critical_prob,
            hysteresis=0.05,
        ))
        
        # RUL (só baixo)
        self.add_threshold(AlarmThreshold(
            param="rul_minutes",
            alarm_type=AlarmType.RUL,
            low_critical=cfg.rul_critical_min,
            low_warning=cfg.rul_warning_min,
            hysteresis=30.0,  # 30 minutos de histerese
        ))
        
        # Health (só baixo)
        self.add_threshold(AlarmThreshold(
            param="health_percent",
            alarm_type=AlarmType.HEALTH,
            low_critical=cfg.health_critical_pct,
            low_warning=cfg.health_warning_pct,
            hysteresis=5.0,
        ))
        
        # Vibração (só alto)
        self.add_threshold(AlarmThreshold(
            param="overall_vibration",
            alarm_type=AlarmType.VIBRATION,
            high_warning=cfg.vibration_warning,
            high_critical=cfg.vibration_critical,
            hysteresis=0.3,
        ))
        
        # Temperatura (só alto)
        self.add_threshold(AlarmThreshold(
            param="motor_temperature",
            alarm_type=AlarmType.TEMPERATURE,
            high_warning=cfg.temperature_warning,
            high_critical=cfg.temperature_critical,
            hysteresis=3.0,
        ))
        
        # Pressão (baixo e alto)
        self.add_threshold(AlarmThreshold(
            param="discharge_pressure",
            alarm_type=AlarmType.PRESSURE,
            low_critical=cfg.pressure_low_critical,
            low_warning=cfg.pressure_low_warning,
            high_warning=cfg.pressure_high_warning,
            high_critical=cfg.pressure_high_critical,
            hysteresis=0.2,
        ))
    
    def add_threshold(self, threshold: AlarmThreshold):
        """Adiciona ou atualiza threshold."""
        self.thresholds[threshold.param] = threshold
    
    def remove_threshold(self, param: str):
        """Remove threshold."""
        if param in self.thresholds:
            del self.thresholds[param]
    
    def _check_value(
        self,
        param: str,
        value: float,
        threshold: AlarmThreshold,
        current_state: Optional[AlarmState]
    ) -> Tuple[AlarmLevel, Optional[str], Optional[float]]:
        """
        Verifica valor contra threshold.
        
        Returns:
            (level, message, threshold_value)
        """
        if not threshold.enabled:
            return AlarmLevel.NONE, None, None
        
        # Aplicar hysteresis se já em alarme
        hyst = threshold.hysteresis if current_state else 0.0
        
        # Check high critical
        if threshold.high_critical is not None:
            limit = threshold.high_critical + (hyst if current_state and current_state.level >= AlarmLevel.CRITICAL else -hyst)
            if value >= limit:
                return AlarmLevel.CRITICAL, f"{param} HIGH CRITICAL: {value:.2f} >= {threshold.high_critical:.2f}", threshold.high_critical
        
        # Check high warning
        if threshold.high_warning is not None:
            limit = threshold.high_warning + (hyst if current_state and current_state.level >= AlarmLevel.WARNING else -hyst)
            if value >= limit:
                return AlarmLevel.WARNING, f"{param} HIGH WARNING: {value:.2f} >= {threshold.high_warning:.2f}", threshold.high_warning
        
        # Check low critical
        if threshold.low_critical is not None:
            limit = threshold.low_critical - (hyst if current_state and current_state.level >= AlarmLevel.CRITICAL else -hyst)
            if value <= limit:
                return AlarmLevel.CRITICAL, f"{param} LOW CRITICAL: {value:.2f} <= {threshold.low_critical:.2f}", threshold.low_critical
        
        # Check low warning
        if threshold.low_warning is not None:
            limit = threshold.low_warning - (hyst if current_state and current_state.level >= AlarmLevel.WARNING else -hyst)
            if value <= limit:
                return AlarmLevel.WARNING, f"{param} LOW WARNING: {value:.2f} <= {threshold.low_warning:.2f}", threshold.low_warning
        
        return AlarmLevel.NONE, None, None
    
    def evaluate(self, values: Dict[str, float]) -> List[AlarmState]:
        """
        Avalia valores contra thresholds.
        
        Args:
            values: Dict com valores dos parâmetros
        
        Returns:
            Lista de alarmes novos ou alterados
        """
        now = datetime.now()
        changed_alarms = []
        
        for param, threshold in self.thresholds.items():
            if param not in values:
                continue
            
            value = values[param]
            current_state = self.active_alarms.get(param)
            
            # Check threshold
            level, message, thresh_val = self._check_value(
                param, value, threshold, current_state
            )
            
            # Se nível mudou
            if level != AlarmLevel.NONE:
                if current_state is None or current_state.level != level:
                    # Novo alarme ou mudança de nível
                    
                    # Verificar tempo mínimo (anti-transiente)
                    pending_key = f"{param}_{level.name}"
                    
                    if pending_key in self._pending_alarms:
                        start_time, pending_level, _ = self._pending_alarms[pending_key]
                        duration = (now - start_time).total_seconds()
                        
                        if duration >= threshold.min_duration_sec:
                            # Confirmar alarme
                            alarm = AlarmState(
                                param=param,
                                level=level,
                                message=message,
                                value=value,
                                threshold=thresh_val,
                                triggered_at=start_time,
                            )
                            self.active_alarms[param] = alarm
                            self.alarm_history.append(alarm)
                            changed_alarms.append(alarm)
                            del self._pending_alarms[pending_key]
                    else:
                        # Iniciar contagem
                        self._pending_alarms[pending_key] = (now, level, value)
                
                else:
                    # Atualizar valor
                    current_state.value = value
            
            else:
                # Alarme cleared
                if current_state is not None and not current_state.cleared:
                    current_state.cleared = True
                    current_state.cleared_at = now
                    changed_alarms.append(current_state)
                    
                    # Mover para histórico após clear
                    del self.active_alarms[param]
                
                # Limpar pendentes
                for key in list(self._pending_alarms.keys()):
                    if key.startswith(param):
                        del self._pending_alarms[key]
        
        return changed_alarms
    
    def acknowledge(self, param: str) -> bool:
        """Acknowledges um alarme."""
        if param in self.active_alarms:
            self.active_alarms[param].acknowledged = True
            self.active_alarms[param].acknowledged_at = datetime.now()
            return True
        return False
    
    def acknowledge_all(self):
        """Acknowledges todos os alarmes ativos."""
        now = datetime.now()
        for alarm in self.active_alarms.values():
            alarm.acknowledged = True
            alarm.acknowledged_at = now
    
    def get_active_alarms(
        self,
        min_level: AlarmLevel = AlarmLevel.WARNING
    ) -> List[AlarmState]:
        """Retorna alarmes ativos acima de um nível."""
        return [
            a for a in self.active_alarms.values()
            if a.level.value >= min_level.value and not a.cleared
        ]
    
    def get_highest_level(self) -> AlarmLevel:
        """Retorna nível mais alto dos alarmes ativos."""
        if not self.active_alarms:
            return AlarmLevel.NONE
        
        return max(
            (a.level for a in self.active_alarms.values() if not a.cleared),
            default=AlarmLevel.NONE
        )
    
    def to_dict(self, alarm: AlarmState) -> Dict:
        """Converte alarme para dict."""
        return {
            "param": alarm.param,
            "level": alarm.level.name,
            "level_value": alarm.level.value,
            "message": alarm.message,
            "value": alarm.value,
            "threshold": alarm.threshold,
            "triggered_at": alarm.triggered_at.isoformat(),
            "acknowledged": alarm.acknowledged,
            "acknowledged_at": alarm.acknowledged_at.isoformat() if alarm.acknowledged_at else None,
            "cleared": alarm.cleared,
            "cleared_at": alarm.cleared_at.isoformat() if alarm.cleared_at else None,
        }
    
    def get_summary(self) -> Dict:
        """Retorna resumo do estado de alarmes."""
        
        active = self.get_active_alarms(AlarmLevel.INFO)
        
        return {
            "total_active": len(active),
            "critical": len([a for a in active if a.level == AlarmLevel.CRITICAL]),
            "warning": len([a for a in active if a.level == AlarmLevel.WARNING]),
            "info": len([a for a in active if a.level == AlarmLevel.INFO]),
            "unacknowledged": len([a for a in active if not a.acknowledged]),
            "highest_level": self.get_highest_level().name,
            "alarms": [self.to_dict(a) for a in active],
        }


# ===========================================================================
# INTEGRATION WITH PREDICTION
# ===========================================================================

def evaluate_prediction_alarms(
    prediction_result,  # PredictionResult from inference_pump
    alarm_manager: AlarmManager
) -> Dict:
    """
    Avalia alarmes a partir de um resultado de predição.
    
    Args:
        prediction_result: Resultado do PumpInference
        alarm_manager: AlarmManager configurado
    
    Returns:
        Dict com resumo de alarmes
    """
    # Extrair valores relevantes
    values = {
        "severity_critical_prob": prediction_result.severity_probs.get("critical", 0),
        "rul_minutes": prediction_result.rul_minutes,
        "health_percent": prediction_result.health_percent,
    }
    
    # Avaliar
    changed = alarm_manager.evaluate(values)
    
    # Retornar resumo
    return alarm_manager.get_summary()


# ===========================================================================
# MAIN (Demo)
# ===========================================================================

def main():
    """Demo de alarmes."""
    
    print("=" * 60)
    print("ALARM LOGIC DEMO")
    print("=" * 60)
    
    # Criar manager
    manager = AlarmManager()
    
    # Simular sequência de valores
    test_values = [
        {"rul_minutes": 600, "health_percent": 80, "severity_critical_prob": 0.1},
        {"rul_minutes": 400, "health_percent": 70, "severity_critical_prob": 0.3},
        {"rul_minutes": 100, "health_percent": 45, "severity_critical_prob": 0.7},  # Warning RUL e Health
        {"rul_minutes": 50, "health_percent": 25, "severity_critical_prob": 0.85},   # Critical tudo
        {"rul_minutes": 600, "health_percent": 80, "severity_critical_prob": 0.1},   # Clear
    ]
    
    import time
    
    for i, values in enumerate(test_values):
        print(f"\n--- Step {i+1} ---")
        print(f"Values: {values}")
        
        # Avaliar (simular tempo para passar min_duration)
        changed = manager.evaluate(values)
        
        # Simular passagem de tempo
        time.sleep(0.1)
        changed = manager.evaluate(values)
        
        # Mostrar estado
        summary = manager.get_summary()
        print(f"Active alarms: {summary['total_active']}")
        print(f"Highest level: {summary['highest_level']}")
        
        for alarm in summary["alarms"]:
            print(f"  - [{alarm['level']}] {alarm['message']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
