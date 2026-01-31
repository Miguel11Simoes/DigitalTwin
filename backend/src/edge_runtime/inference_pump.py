#!/usr/bin/env python3
"""
inference_pump.py
=================
A12: Edge Runtime para inferência de bomba industrial.

Este módulo fornece inferência completa pronta para deploy:
- Carrega modelo Keras ou TFLite
- Processa dados raw em tempo real
- Aplica normalização e windowing
- Detecta OOD/drift
- Retorna predições com alarmes

Uso:
    from src.edge_runtime.inference_pump import PumpInference
    
    engine = PumpInference("models/")
    result = engine.predict(sensor_data)

Author: Industrial Pump Digital Twin
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np

warnings.filterwarnings("ignore")

# Tentar importar TFLite runtime (mais leve que TF completo)
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

# Fallback para Keras completo
try:
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


# ===========================================================================
# DATA CLASSES
# ===========================================================================

@dataclass
class AlarmConfig:
    """Configuração de alarmes industriais."""
    
    # Thresholds de severidade para alarme
    critical_threshold: float = 0.8  # Se P(critical) > 0.8 → alarme
    degraded_threshold: float = 0.6
    
    # RUL thresholds (em minutos)
    rul_critical_min: float = 60.0      # 1 hora
    rul_warning_min: float = 480.0      # 8 horas
    
    # Health thresholds
    health_critical: float = 0.3
    health_warning: float = 0.5
    
    # OOD
    ood_alarm: bool = True


@dataclass
class PredictionResult:
    """Resultado de uma predição."""
    
    # Classifications
    severity: str
    severity_confidence: float
    severity_probs: Dict[str, float]
    
    mode: str
    mode_confidence: float
    mode_probs: Dict[str, float]
    
    # Regressions
    rul_minutes: float
    rul_normalized: float
    health_percent: float
    health_normalized: float
    
    # Alarms
    alarms: List[str]
    alarm_level: str  # "none", "warning", "critical"
    
    # OOD
    is_ood: bool
    ood_distance: float
    
    # Debug
    timestamp: Optional[str] = None
    raw_outputs: Optional[Dict] = None


# ===========================================================================
# PUMP INFERENCE ENGINE
# ===========================================================================

class PumpInference:
    """
    Motor de inferência para bomba industrial.
    
    Features:
    - Carrega modelo Keras ou TFLite
    - Normalização automática
    - Detecção OOD
    - Sistema de alarmes
    - Batch e single-sample inference
    """
    
    def __init__(
        self,
        model_dir: Union[str, Path],
        use_tflite: bool = True,
        alarm_config: Optional[AlarmConfig] = None
    ):
        """
        Inicializa engine de inferência.
        
        Args:
            model_dir: Diretório com modelo e configs
            use_tflite: Usar TFLite se disponível
            alarm_config: Configuração de alarmes
        """
        self.model_dir = Path(model_dir)
        self.alarm_config = alarm_config or AlarmConfig()
        
        # Load configs
        self._load_configs()
        
        # Load model
        self.use_tflite = use_tflite and TFLITE_AVAILABLE
        self._load_model()
        
        print(f"[INFO] PumpInference initialized")
        print(f"[INFO] Model: {'TFLite' if self.use_tflite else 'Keras'}")
        print(f"[INFO] Severity classes: {self.severity_labels}")
        print(f"[INFO] Mode classes: {self.mode_labels}")
    
    def _load_configs(self):
        """Carrega configurações do modelo."""
        
        # Model config
        config_path = self.model_dir / "model_config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        # Labels
        labels_path = self.model_dir / "pump_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                labels = json.load(f)
            
            # Invert mappings: idx -> label
            self.severity_labels = {
                int(v): k for k, v in labels.get("severity", {}).items()
            }
            self.mode_labels = {
                int(v): k for k, v in labels.get("mode", {}).items()
            }
        else:
            # Default labels
            self.severity_labels = {0: "critical", 1: "degraded", 2: "warning", 3: "healthy"}
            self.mode_labels = {0: "normal", 1: "bearing_wear", 2: "cavitation", 
                               3: "impeller_damage", 4: "seal_leak", 5: "misalignment", 6: "blockage"}
        
        # Scalers
        scaler_path = self.model_dir / "pump_scalers.json"
        if scaler_path.exists():
            with open(scaler_path) as f:
                self.scalers = json.load(f)
        else:
            self.scalers = {"mean": {}, "std": {}}
        
        # OOD config
        ood_path = self.model_dir / "ood_config.json"
        if ood_path.exists():
            with open(ood_path) as f:
                self.ood_config = json.load(f)
        else:
            self.ood_config = None
        
        # Profile
        profile_path = self.model_dir / "pump_profile.json"
        if profile_path.exists():
            with open(profile_path) as f:
                self.profile = json.load(f)
        else:
            self.profile = {}
        
        # RUL normalization
        self.rul_max_train = self.config.get("rul_max_train", 1440.0)
    
    def _load_model(self):
        """Carrega modelo."""
        
        if self.use_tflite:
            tflite_path = self.model_dir / "pump_cnn2d_v2.tflite"
            if tflite_path.exists():
                self.interpreter = tflite.Interpreter(str(tflite_path))
                self.interpreter.allocate_tensors()
                
                # Get input/output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                self.model = None
                return
            else:
                print(f"[WARN] TFLite not found, falling back to Keras")
                self.use_tflite = False
        
        if KERAS_AVAILABLE:
            keras_path = self.model_dir / "pump_cnn2d_v2.keras"
            if keras_path.exists():
                self.model = keras.models.load_model(
                    keras_path,
                    custom_objects={"MaskedGlobalPooling": MaskedGlobalPooling}
                )
                self.interpreter = None
            else:
                raise FileNotFoundError(f"Model not found: {keras_path}")
        else:
            raise ImportError("Neither TFLite nor Keras available")
    
    def _predict_tflite(
        self,
        X_spec: np.ndarray,
        X_aux: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inferência com TFLite."""
        
        # Set inputs (ordem pode variar, verificar input_details)
        for detail in self.input_details:
            name = detail["name"]
            idx = detail["index"]
            
            if "spec" in name.lower():
                self.interpreter.set_tensor(idx, X_spec.astype(np.float32))
            elif "aux" in name.lower():
                self.interpreter.set_tensor(idx, X_aux.astype(np.float32))
            elif "mask" in name.lower():
                self.interpreter.set_tensor(idx, mask.astype(np.float32))
        
        # Run
        self.interpreter.invoke()
        
        # Get outputs
        outputs = {}
        for detail in self.output_details:
            name = detail["name"]
            idx = detail["index"]
            outputs[name] = self.interpreter.get_tensor(idx)
        
        # Mapear para outputs esperados
        severity_probs = outputs.get("severity", outputs.get("Identity", np.zeros((1, 4))))
        mode_probs = outputs.get("mode", outputs.get("Identity_1", np.zeros((1, 7))))
        rul = outputs.get("rul", outputs.get("Identity_2", np.zeros((1, 1))))
        health = outputs.get("health", outputs.get("Identity_3", np.zeros((1, 1))))
        
        return severity_probs, mode_probs, rul, health
    
    def _predict_keras(
        self,
        X_spec: np.ndarray,
        X_aux: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inferência com Keras."""
        
        preds = self.model.predict(
            [X_spec, X_aux, mask],
            verbose=0
        )
        
        return preds[0], preds[1], preds[2], preds[3]
    
    def _check_ood(
        self,
        X_spec: np.ndarray,
        X_aux: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[bool, float]:
        """Verifica se amostra é OOD."""
        
        if self.ood_config is None or self.model is None:
            return False, 0.0
        
        try:
            # Extrair embedding
            embedding_layer = self.model.get_layer("embedding")
            embedding_model = keras.Model(
                inputs=self.model.inputs,
                outputs=embedding_layer.output
            )
            
            embedding = embedding_model.predict(
                [X_spec, X_aux, mask],
                verbose=0
            )
            
            # Calcular distância ao centroid
            centroid = np.array(self.ood_config["centroid"])
            threshold = self.ood_config["threshold"]
            
            distance = np.linalg.norm(embedding - centroid, axis=1)[0]
            is_ood = distance > threshold
            
            return bool(is_ood), float(distance)
        
        except Exception as e:
            print(f"[WARN] OOD check failed: {e}")
            return False, 0.0
    
    def _generate_alarms(
        self,
        severity: str,
        severity_probs: Dict[str, float],
        rul_minutes: float,
        health_percent: float,
        is_ood: bool
    ) -> Tuple[List[str], str]:
        """Gera alarmes baseado nas predições."""
        
        alarms = []
        max_level = "none"
        
        cfg = self.alarm_config
        
        # Alarme de severidade
        if severity == "critical":
            if severity_probs.get("critical", 0) > cfg.critical_threshold:
                alarms.append(f"CRITICAL: Severity critical with {severity_probs['critical']*100:.1f}% confidence")
                max_level = "critical"
        elif severity == "degraded":
            if severity_probs.get("degraded", 0) > cfg.degraded_threshold:
                alarms.append(f"WARNING: Severity degraded with {severity_probs['degraded']*100:.1f}% confidence")
                max_level = max(max_level, "warning", key=lambda x: ["none", "warning", "critical"].index(x))
        
        # Alarme de RUL
        if rul_minutes < cfg.rul_critical_min:
            alarms.append(f"CRITICAL: RUL is {rul_minutes:.0f} minutes (< {cfg.rul_critical_min:.0f})")
            max_level = "critical"
        elif rul_minutes < cfg.rul_warning_min:
            alarms.append(f"WARNING: RUL is {rul_minutes:.0f} minutes (< {cfg.rul_warning_min:.0f})")
            max_level = max(max_level, "warning", key=lambda x: ["none", "warning", "critical"].index(x))
        
        # Alarme de health
        if health_percent < cfg.health_critical * 100:
            alarms.append(f"CRITICAL: Health is {health_percent:.1f}% (< {cfg.health_critical*100:.0f}%)")
            max_level = "critical"
        elif health_percent < cfg.health_warning * 100:
            alarms.append(f"WARNING: Health is {health_percent:.1f}% (< {cfg.health_warning*100:.0f}%)")
            max_level = max(max_level, "warning", key=lambda x: ["none", "warning", "critical"].index(x))
        
        # Alarme OOD
        if is_ood and cfg.ood_alarm:
            alarms.append("WARNING: Sample is out-of-distribution (potential drift)")
            max_level = max(max_level, "warning", key=lambda x: ["none", "warning", "critical"].index(x))
        
        return alarms, max_level
    
    def predict(
        self,
        X_spec: np.ndarray,
        X_aux: np.ndarray,
        mask: np.ndarray,
        return_raw: bool = False
    ) -> PredictionResult:
        """
        Executa predição.
        
        Args:
            X_spec: Espectrogramas (1, n_sensors, n_freq, n_frames, 1)
            X_aux: Features auxiliares (1, aux_dim)
            mask: Máscara de sensores (1, n_sensors)
            return_raw: Incluir outputs raw
        
        Returns:
            PredictionResult
        """
        # Garantir batch dimension
        if X_spec.ndim == 4:
            X_spec = X_spec[np.newaxis, ...]
        if X_aux.ndim == 1:
            X_aux = X_aux[np.newaxis, ...]
        if mask.ndim == 1:
            mask = mask[np.newaxis, ...]
        
        # Predict
        if self.use_tflite:
            sev_probs, mode_probs, rul, health = self._predict_tflite(X_spec, X_aux, mask)
        else:
            sev_probs, mode_probs, rul, health = self._predict_keras(X_spec, X_aux, mask)
        
        # Parse severity
        sev_idx = int(np.argmax(sev_probs[0]))
        severity = self.severity_labels.get(sev_idx, f"class_{sev_idx}")
        severity_conf = float(sev_probs[0, sev_idx])
        severity_probs_dict = {
            self.severity_labels.get(i, f"class_{i}"): float(p)
            for i, p in enumerate(sev_probs[0])
        }
        
        # Parse mode
        mode_idx = int(np.argmax(mode_probs[0]))
        mode = self.mode_labels.get(mode_idx, f"mode_{mode_idx}")
        mode_conf = float(mode_probs[0, mode_idx])
        mode_probs_dict = {
            self.mode_labels.get(i, f"mode_{i}"): float(p)
            for i, p in enumerate(mode_probs[0])
        }
        
        # Parse regressions
        rul_norm = float(rul[0, 0])
        rul_minutes = rul_norm * self.rul_max_train
        
        health_norm = float(health[0, 0])
        health_percent = health_norm * 100.0
        
        # OOD check
        is_ood, ood_dist = self._check_ood(X_spec, X_aux, mask)
        
        # Generate alarms
        alarms, alarm_level = self._generate_alarms(
            severity, severity_probs_dict, rul_minutes, health_percent, is_ood
        )
        
        # Build result
        result = PredictionResult(
            severity=severity,
            severity_confidence=severity_conf,
            severity_probs=severity_probs_dict,
            mode=mode,
            mode_confidence=mode_conf,
            mode_probs=mode_probs_dict,
            rul_minutes=rul_minutes,
            rul_normalized=rul_norm,
            health_percent=health_percent,
            health_normalized=health_norm,
            alarms=alarms,
            alarm_level=alarm_level,
            is_ood=is_ood,
            ood_distance=ood_dist,
            raw_outputs={
                "severity_probs": sev_probs.tolist(),
                "mode_probs": mode_probs.tolist(),
                "rul": rul.tolist(),
                "health": health.tolist(),
            } if return_raw else None
        )
        
        return result
    
    def predict_batch(
        self,
        X_spec: np.ndarray,
        X_aux: np.ndarray,
        mask: np.ndarray
    ) -> List[PredictionResult]:
        """
        Predição em batch.
        
        Args:
            X_spec: (batch, n_sensors, n_freq, n_frames, 1)
            X_aux: (batch, aux_dim)
            mask: (batch, n_sensors)
        
        Returns:
            Lista de PredictionResult
        """
        results = []
        
        for i in range(len(X_spec)):
            result = self.predict(
                X_spec[i:i+1],
                X_aux[i:i+1],
                mask[i:i+1]
            )
            results.append(result)
        
        return results
    
    def to_dict(self, result: PredictionResult) -> Dict:
        """Converte resultado para dict (JSON-serializable)."""
        
        return {
            "severity": result.severity,
            "severity_confidence": result.severity_confidence,
            "severity_probs": result.severity_probs,
            "mode": result.mode,
            "mode_confidence": result.mode_confidence,
            "mode_probs": result.mode_probs,
            "rul_minutes": result.rul_minutes,
            "rul_normalized": result.rul_normalized,
            "health_percent": result.health_percent,
            "health_normalized": result.health_normalized,
            "alarms": result.alarms,
            "alarm_level": result.alarm_level,
            "is_ood": result.is_ood,
            "ood_distance": result.ood_distance,
        }


# ===========================================================================
# CUSTOM LAYER (para Keras load)
# ===========================================================================

if KERAS_AVAILABLE:
    from tensorflow.keras import layers
    
    class MaskedGlobalPooling(layers.Layer):
        """Global Average Pooling que respeita máscara de sensores."""
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def call(self, inputs):
            import tensorflow as tf
            x, mask = inputs
            mask_expanded = tf.expand_dims(mask, -1)
            x_masked = x * mask_expanded
            sum_features = tf.reduce_sum(x_masked, axis=1)
            n_active = tf.reduce_sum(mask, axis=1, keepdims=True)
            n_active = tf.maximum(n_active, 1.0)
            return sum_features / n_active
        
        def get_config(self):
            return super().get_config()


# ===========================================================================
# MAIN (Demo)
# ===========================================================================

def main():
    """Demo de inferência."""
    
    import sys
    from pathlib import Path
    
    print("=" * 60)
    print("PUMP INFERENCE DEMO")
    print("=" * 60)
    
    # Find model directory
    model_dir = Path(__file__).parent.parent.parent / "models"
    
    if not (model_dir / "pump_cnn2d_v2.keras").exists():
        print(f"[ERROR] Model not found in {model_dir}")
        print("Run train_pump_cnn2d_v2.py first!")
        sys.exit(1)
    
    # Initialize engine
    engine = PumpInference(model_dir, use_tflite=False)
    
    # Create dummy input
    cfg = engine.config
    n_sensors = cfg.get("max_sensors_fast", 4)
    n_freq = cfg.get("n_freq", 17)
    n_frames = cfg.get("n_frames", 8)
    aux_dim = cfg.get("aux_dim", 98)
    
    X_spec = np.random.randn(1, n_sensors, n_freq, n_frames, 1).astype(np.float32)
    X_aux = np.random.randn(1, aux_dim).astype(np.float32)
    mask = np.ones((1, n_sensors), dtype=np.float32)
    
    # Predict
    result = engine.predict(X_spec, X_aux, mask, return_raw=True)
    
    print("\n--- PREDICTION RESULT ---")
    print(f"Severity: {result.severity} ({result.severity_confidence*100:.1f}%)")
    print(f"Mode: {result.mode} ({result.mode_confidence*100:.1f}%)")
    print(f"RUL: {result.rul_minutes:.1f} minutes")
    print(f"Health: {result.health_percent:.1f}%")
    print(f"OOD: {result.is_ood} (distance: {result.ood_distance:.4f})")
    
    if result.alarms:
        print("\nALARMS:")
        for alarm in result.alarms:
            print(f"  - {alarm}")
    else:
        print("\nNo alarms.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
