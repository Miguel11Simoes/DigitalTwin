# Backend Structure - Digital Twin Pump Predictive Maintenance

## 📁 Estrutura Organizada

```
backend/
├── training/           # Scripts de treino de modelos
│   ├── train_cnn_simple.py     # CNN 1D simplificado (ALL TARGETS ✓)
│   ├── train_cnn_2d.py         # CNN 2D produto-ready (espectrogramas)
│   ├── train_fast.py           # Sklearn baseline
│   ├── train_pump_predictive_market.py  # Pipeline completo
│   └── ...
│
├── generators/         # Geradores de datasets sintéticos
│   ├── generate_dataset_v2.py  # Non-overlapping severity (RECOMENDADO)
│   ├── generate_dataset_v3.py  # Temporal por asset
│   └── ...
│
├── datasets/           # Datasets CSV gerados
│   ├── sensors_log_v2.csv      # Dataset principal
│   └── ...
│
├── utils/              # Funções auxiliares
│   ├── focal_loss.py
│   ├── evaluate_report.py
│   └── ...
│
├── scripts/            # Scripts batch e shell
│   ├── run_preset_*.bat
│   └── monitor_*.sh
│
├── outputs/            # Outputs de treino
│   ├── logs/           # Logs de execução
│   └── reports/        # Relatórios JSON
│
├── models/             # Modelos treinados e artefactos
│   ├── pump_cnn_simple.keras
│   ├── eval_report_cnn.json
│   └── ...
│
├── logs/               # Logs operacionais (streaming)
│   ├── sensors_log.csv
│   └── vibration_waveform/
│
├── docs/               # Documentação
│
└── main.py             # API Flask principal
```

## 🎯 Modelos com Melhores Resultados

### 1. CNN Simplificado 1D (`train_cnn_simple.py`)
- **RUL MAE**: 1.64% ✓
- **Health MAE**: 1.45% ✓
- **Severity acc**: 95.41% ✓
- **Mode acc**: 100.00% ✓
- **TARGETS MET: 4/4**

### 2. Sklearn Baseline (`train_fast.py`)
- Severity: 98.17%
- Mode: 100%
- RUL MAE: 4%
- Health MAE: 2.45%

## 🔧 Como Usar

```bash
# Gerar dataset
python generators/generate_dataset_v2.py

# Treinar modelo CNN 1D
python training/train_cnn_simple.py

# Treinar modelo CNN 2D produto-ready
python training/train_cnn_2d.py
```

## ⚠️ Notas Importantes

1. **Nunca apagar ficheiros originais** - sempre copiar
2. **Split por asset_id** - obrigatório para evitar leakage
3. **Validar baselines sklearn** antes de deep learning
