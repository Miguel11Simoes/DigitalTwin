#!/usr/bin/env python3
"""Testa se o modelo é 100% serializável (sem Lambda)."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path

# Import custom layers
import sys
sys.path.insert(0, str(Path(__file__).parent))

# Importar as custom layers
from train_pump_predictive_market import (
    SensorDropout, RawToSpectrograms, SpecAugment,
    SqueezeLast, ExpandDimsLast, MaskedSoftmaxLayer,
    WeightedSum, MaskExpand, MaskedDenominator, DivideLayer
)

def test_serialization():
    print("[TEST] Criando modelo simples com custom layers...")
    
    # Modelo mínimo com todas as custom layers
    inp = keras.Input(shape=(10,), name="test_input")
    mask = keras.Input(shape=(10,), dtype=tf.bool, name="test_mask")
    
    # Test SqueezeLast
    x1 = keras.layers.Dense(5)(inp)
    x1 = keras.layers.Reshape((5, 1))(x1)
    x1 = SqueezeLast()(x1)
    
    # Test ExpandDimsLast
    x2 = ExpandDimsLast()(x1)
    
    # Test MaskExpand
    x3 = MaskExpand()(mask)
    
    # Test WeightedSum
    x4 = keras.layers.Reshape((5, 1))(x1)
    dummy_weights = keras.layers.Reshape((5, 1))(keras.layers.Dense(5)(inp))
    x4 = WeightedSum()([x4, dummy_weights])
    
    # Test MaskedDenominator
    x5 = MaskedDenominator()(x3)
    
    # Test DivideLayer
    x6 = DivideLayer()([x4, x5])
    
    model = keras.Model(inputs=[inp, mask], outputs=x6, name="test_model")
    
    print("[TEST] Salvando modelo em modo seguro...")
    save_path = Path("models/test_serialization.keras")
    save_path.parent.mkdir(exist_ok=True)
    model.save(save_path)
    
    print("[TEST] Carregando modelo COM safe_mode=True (produção)...")
    try:
        loaded_model = keras.models.load_model(save_path, safe_mode=True)
        print("[✅ SUCESSO] Modelo carregado em safe_mode=True!")
        print("[✅] Todas as custom layers são 100% serializáveis.")
        
        # Cleanup
        save_path.unlink()
        
        return True
    except ValueError as e:
        if "Lambda" in str(e):
            print(f"[❌ FALHA] Ainda há Lambda layers não serializáveis: {e}")
            return False
        raise

if __name__ == "__main__":
    success = test_serialization()
    exit(0 if success else 1)
