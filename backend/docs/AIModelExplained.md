# Modelo de Inteligência Artificial para Manutenção Preditiva de Bombas Industriais

## Relatório Técnico Completo para Tese de Mestrado

**Autor:** Miguel Simões  
**Data:** Janeiro 2026  
**Projeto:** Digital Twin - Sistema de Manutenção Preditiva  
**Repositório:** github.com/Miguel11Simoes/DigitalTwin

---

## Índice

1. [Introdução e Objetivo](#1-introdução-e-objetivo)
2. [Arquitetura do Sistema](#2-arquitetura-do-sistema)
3. [Geração do Dataset](#3-geração-do-dataset)
4. [Pré-processamento de Dados](#4-pré-processamento-de-dados)
5. [Arquitetura do Modelo CNN](#5-arquitetura-do-modelo-cnn)
6. [Camadas Personalizadas](#6-camadas-personalizadas)
7. [Pipeline de Treino](#7-pipeline-de-treino)
8. [Decisões de Design](#8-decisões-de-design)
9. [Resultados Obtidos](#9-resultados-obtidos)
10. [Validação e Testes de Stress](#10-validação-e-testes-de-stress)
11. [Conclusões](#11-conclusões)
12. [Referências Técnicas](#12-referências-técnicas)

---

## 1. Introdução e Objetivo

### 1.1 Contexto Industrial

A manutenção preditiva representa uma evolução fundamental na gestão de ativos industriais. Ao contrário da manutenção reativa (que atua após a falha) ou preventiva (baseada em intervalos fixos), a manutenção preditiva utiliza dados de sensores e algoritmos de Machine Learning para prever quando um equipamento irá falhar, permitindo intervenções no momento ideal.

### 1.2 Objetivo do Projeto

O objetivo principal deste projeto é desenvolver um **modelo de Deep Learning** capaz de:

1. **Classificar a severidade** do estado de degradação de bombas industriais (5 classes: normal, early, moderate, severe, failure)
2. **Identificar o modo de falha** (5 modos: normal_operation, bearing_wear, cavitation, misalignment, imbalance)
3. **Estimar o RUL** (Remaining Useful Life) - tempo restante até falha
4. **Calcular o Health Index** - percentagem de "saúde" do equipamento

### 1.3 Targets Obrigatórios

Os requisitos mínimos para aprovação do modelo foram definidos como:

| Métrica | Target | Justificação |
|---------|--------|--------------|
| **Severity Accuracy** | > 90% | Crítico para decisões de manutenção |
| **Mode Accuracy** | > 90% | Necessário para diagnóstico correto |
| **RUL MAE** | < 20% | Margem aceitável para planeamento |
| **Health MAE** | < 10% | Precisão necessária para dashboards |

---

## 2. Arquitetura do Sistema

### 2.1 Estrutura de Ficheiros

```
backend/
├── training/                    # Scripts de treino
│   ├── train_cnn_2d.py         # Pipeline CNN 2D principal (PRODUTO)
│   ├── train_cnn_simple.py     # Pipeline CNN 1D simplificado
│   └── ...
├── generators/                  # Geradores de datasets
│   ├── generate_dataset_v2.py  # Dataset v2 (usado em produção)
│   └── ...
├── datasets/                    # Datasets CSV
│   └── sensors_log_v2.csv      # 80,000 amostras
├── utils/                       # Utilitários
│   ├── focal_loss.py           # Focal Loss implementation
│   └── evaluate_report.py      # Funções de avaliação
├── models/                      # Modelos treinados
│   └── pump_cnn_2d_product.keras
└── outputs/reports/             # Relatórios JSON
    └── eval_report_cnn_2d.json
```

### 2.2 Tecnologias Utilizadas

- **TensorFlow/Keras 3.x** - Framework de Deep Learning
- **scikit-learn** - Baseline e métricas
- **NumPy/Pandas** - Manipulação de dados
- **Python 3.10+** - Linguagem de programação

---

## 3. Geração do Dataset

### 3.1 O Problema Inicial

O primeiro dataset gerado (`sensors_log.csv`) apresentava um problema crítico: **sobreposição total dos ranges de severidade**. Isto significava que amostras de classes diferentes tinham valores de sensores idênticos, tornando a classificação impossível.

```
# Problema original (sobreposição 100%)
normal:   vibration 0.0-5.0
early:    vibration 0.0-5.0  ← IGUAL!
moderate: vibration 0.0-5.0  ← IGUAL!
```

### 3.2 Solução: Dataset v2 com Ranges Não-Sobrepostos

O ficheiro `generators/generate_dataset_v2.py` implementa a geração corrigida:

```python
# Ranges DISTINTOS para cada severity (com gaps de 2%)
if severity_choice == "normal":
    degradation = rng.uniform(0.00, 0.10)  # 0-10%
    health_index = rng.uniform(90, 100)
elif severity_choice == "early":
    degradation = rng.uniform(0.12, 0.25)  # 12-25% (gap de 2%)
    health_index = rng.uniform(75, 88)
elif severity_choice == "moderate":
    degradation = rng.uniform(0.27, 0.50)  # 27-50%
    health_index = rng.uniform(50, 73)
elif severity_choice == "severe":
    degradation = rng.uniform(0.52, 0.75)  # 52-75%
    health_index = rng.uniform(25, 48)
else:  # failure
    degradation = rng.uniform(0.77, 1.00)  # 77-100%
    health_index = rng.uniform(0, 23)
```

### 3.3 Assinaturas Distintas por Modo de Falha

Cada modo de falha tem uma "assinatura" única nos sensores:

| Modo | Vibração X | Vibração Y | Vibração Z | Temperatura | Pressão |
|------|------------|------------|------------|-------------|---------|
| **normal_operation** | 35% | 35% | 30% | Normal | Normal |
| **bearing_wear** | **60%** | 25% | 15% | **+8°C** | -0.1 |
| **cavitation** | 25% | **55%** | 20% | +3°C | **-0.4** |
| **misalignment** | 45% | 45% | 10% | +5°C | -0.2 |
| **imbalance** | 20% | 20% | **60%** | +4°C | -0.15 |

### 3.4 Sensores Simulados

O dataset inclui 8 sensores:

```python
SENSOR_NAMES = [
    "overall_vibration",  # Vibração total (mm/s)
    "vibration_x",        # Componente X
    "vibration_y",        # Componente Y
    "vibration_z",        # Componente Z
    "motor_current",      # Corrente do motor (A)
    "pressure",           # Pressão (bar)
    "flow",               # Caudal (L/min)
    "temperature"         # Temperatura (°C)
]
```

### 3.5 Estatísticas do Dataset Final

- **Total de amostras:** 80,000
- **Assets (bombas):** 8
- **Amostras por asset:** 10,000
- **Distribuição de classes:** Balanceada (20% cada)

---

## 4. Pré-processamento de Dados

### 4.1 Split Anti-Leakage por Asset

**Decisão crítica:** O split de dados é feito por `asset_id` completo, não por amostras aleatórias. Isto previne data leakage temporal.

```python
def split_by_asset(self, df: pd.DataFrame):
    """
    Split data by asset_id (ANTI-LEAKAGE OBRIGATÓRIO).
    Assets inteiros vão para train, val ou test.
    """
    assets = df['asset_id'].unique()
    n_assets = len(assets)
    
    # Shuffle assets
    np.random.shuffle(assets)
    
    # 70% train, 15% val, 15% test
    n_train = int(0.7 * n_assets)
    n_val = int(0.15 * n_assets)
    
    train_assets = assets[:n_train]
    val_assets = assets[n_train:n_train + n_val]
    test_assets = assets[n_train + n_val:]
```

**Justificação:** Se amostras do mesmo asset aparecerem em train e test, o modelo pode memorizar padrões específicos desse asset, inflacionando artificialmente as métricas.

### 4.2 Normalização

```python
# StandardScaler para sensores
self.scaler = StandardScaler()
df_norm[sensor_cols] = self.scaler.fit_transform(df[sensor_cols])

# Normalização de targets
df['rul_norm'] = df['rul_minutes'] / rul_max  # 0-1
df['health_norm'] = df['health_index'] / 100.0  # 0-1
```

### 4.3 Windowing Temporal

Os dados são transformados em janelas deslizantes para capturar padrões temporais:

```python
@dataclass
class ProductConfig:
    seq_len: int = 64    # Tamanho da janela (64 timesteps)
    hop: int = 8         # Sobreposição (hop de 8)
```

**Cálculo do número de janelas:**
```
n_windows = (n_samples - seq_len) / hop + 1
```

Para 10,000 amostras por asset: `(10000 - 64) / 8 + 1 ≈ 1,242 janelas`

### 4.4 Criação de Pseudo-Espectrogramas

Para o modelo CNN 2D, os sinais temporais são convertidos em representações 2D:

```python
def _create_pseudo_spectrogram(self, sensor_data: np.ndarray):
    """
    Cria pseudo-espectrograma via FFT por frames.
    Shape final: (max_sensors, n_freq, n_time, 1)
    """
    for s in range(n_sensors):
        signal = sensor_data[s]
        
        # Dividir em frames
        for t in range(n_frames):
            frame = signal[start:end]
            
            # FFT do frame
            fft = np.abs(np.fft.rfft(frame, n=n_freq * 2))[:n_freq]
            spec[s, :, t] = fft
    
    return spec[..., np.newaxis]  # Canal adicional
```

---

## 5. Arquitetura do Modelo CNN

### 5.1 Visão Geral

O modelo `pump_cnn_2d_product` segue uma arquitetura multi-input, multi-output:

```
┌─────────────────────────────────────────────────────────────┐
│                         INPUTS                              │
├─────────────────┬───────────────────┬───────────────────────┤
│ spectrogram     │ sensor_mask       │ aux_features          │
│ (8, 16, 16, 1)  │ (8,)              │ (8,)                  │
└────────┬────────┴────────┬──────────┴───────────┬───────────┘
         │                 │                      │
         ▼                 ▼                      │
┌────────────────┐  ┌──────────────┐              │
│ Per-Sensor     │  │ Sensor       │              │
│ CNN 2D Encoder │  │ Dropout      │              │
│ (shared)       │  │ (training)   │              │
└────────┬───────┘  └──────┬───────┘              │
         │                 │                      │
         ▼                 ▼                      │
┌────────────────────────────────────┐            │
│    Masked Attention Pooling        │            │
│    + Masked Mean Pooling           │            │
└────────────────┬───────────────────┘            │
                 │                                │
                 ▼                                ▼
         ┌───────────────────────────────────────┐
         │           Concatenation               │
         │      (attention + mean + aux)         │
         └───────────────────┬───────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   Dense Layers        │
                 │   + BatchNorm         │
                 │   + Dropout           │
                 └───────────┬───────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Severity    │    │ Mode        │    │ RUL/Health  │
│ (softmax 5) │    │ (softmax 5) │    │ (sigmoid 1) │
└─────────────┘    └─────────────┘    └─────────────┘
```

### 5.2 Per-Sensor CNN 2D Encoder

Cada sensor tem a sua própria representação 2D (espectrograma) que é processada por um encoder CNN partilhado:

```python
# Shared CNN encoder para cada sensor
cnn_input = keras.Input(shape=(cfg.n_mels, cfg.seq_len // 4, 1))

x = cnn_input
for i, filters in enumerate(cfg.cnn_filters):  # [32, 64, 128]
    x = layers.Conv2D(
        filters, (3, 3), padding='same', activation='relu',
        kernel_regularizer=keras.regularizers.l2(cfg.l2_reg)
    )(x)
    if i < len(cfg.cnn_filters) - 1:
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Dropout(cfg.dropout * 0.5)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(cfg.dense_units, activation='relu')(x)

per_sensor_encoder = keras.Model(cnn_input, x, name="per_sensor_encoder")
```

**Parâmetros CNN:**
- Filtros: [32, 64, 128]
- Kernel: 3×3
- Regularização L2: 1e-5
- Dropout: 15% após cada conv

### 5.3 Aplicação a Cada Sensor

```python
# Aplicar encoder a cada sensor
sensor_features_list = []
for s in range(cfg.max_sensors):
    # Extrair espectrograma do sensor s
    sensor_spec = layers.Lambda(lambda x, s=s: x[:, s, :, :, :])(spec_in)
    sensor_feat = per_sensor_encoder(sensor_spec)
    sensor_features_list.append(sensor_feat)

# Stack: (batch, max_sensors, dense_units)
sensor_features = layers.Lambda(
    lambda x: keras.ops.stack(x, axis=1)
)(sensor_features_list)
```

### 5.4 Output Heads

O modelo tem 4 outputs independentes:

```python
# Severity (classificação 5 classes)
sev_h = layers.Dense(64, activation='relu')(fused)
sev_h = layers.Dropout(cfg.dropout)(sev_h)
severity_out = layers.Dense(n_severity, activation='softmax', name='severity')(sev_h)

# Mode (classificação 5 classes)
mode_h = layers.Dense(64, activation='relu')(fused)
mode_h = layers.Dropout(cfg.dropout)(mode_h)
mode_out = layers.Dense(n_mode, activation='softmax', name='mode')(mode_h)

# RUL (regressão 0-1)
rul_h = layers.Dense(64, activation='relu')(fused)
rul_out = layers.Dense(1, activation='sigmoid', name='rul')(rul_h)

# Health (regressão 0-1)
health_h = layers.Dense(64, activation='relu')(fused)
health_out = layers.Dense(1, activation='sigmoid', name='health')(health_h)
```

### 5.5 Parâmetros do Modelo

```
Total params: 169,421 (661.80 KB)
Trainable params: 169,165 (660.80 KB)
Non-trainable params: 256 (1.00 KB)
```

---

## 6. Camadas Personalizadas

### 6.1 SensorDropout

**Propósito:** Simular falhas de sensores durante o treino para aumentar a robustez.

```python
@keras.utils.register_keras_serializable(package="Product")
class SensorDropout(layers.Layer):
    """
    Randomly drops entire sensors during training.
    Simulates sensor failures in production.
    """
    def __init__(self, rate: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, mask, training=None):
        if not training or self.rate <= 0:
            return mask
        
        shape = keras.ops.shape(mask)
        keep = keras.ops.cast(
            keras.random.uniform(shape) >= self.rate, 
            "float32"
        )
        dropped = mask * keep
        
        # NUNCA desligar TODOS os sensores
        all_zero = keras.ops.all(
            keras.ops.equal(dropped, 0.0), 
            axis=-1, keepdims=True
        )
        return keras.ops.where(all_zero, mask, dropped)
```

**Decisão de design:** A taxa de dropout é 15% durante treino. Isto força o modelo a não depender excessivamente de nenhum sensor específico.

### 6.2 MaskedAttentionPooling

**Propósito:** Agregar features de múltiplos sensores usando atenção, respeitando a máscara de sensores válidos.

```python
@keras.utils.register_keras_serializable(package="Product")
class MaskedAttentionPooling(layers.Layer):
    """Attention pooling with sensor mask support."""
    
    def build(self, input_shape):
        feature_dim = input_shape[0][-1]
        self.attn_dense = layers.Dense(1)
    
    def call(self, inputs):
        features, mask = inputs  # (B, S, D), (B, S)
        
        # Calcular scores de atenção
        scores = keras.ops.squeeze(self.attn_dense(features), axis=-1)
        
        # Mascarar sensores inválidos com valor muito negativo
        mask_bool = keras.ops.cast(mask, "bool")
        scores = keras.ops.where(
            mask_bool, scores, 
            keras.ops.full_like(scores, -1e9)
        )
        
        # Softmax normalizado
        weights = keras.ops.softmax(scores, axis=-1)
        weights = keras.ops.expand_dims(weights, -1)
        
        # Pooling ponderado
        pooled = keras.ops.sum(features * weights, axis=1)
        return pooled
```

**Vantagem:** O mecanismo de atenção aprende quais sensores são mais informativos para cada previsão.

### 6.3 MaskedMeanPooling

**Propósito:** Pooling simples por média, respeitando a máscara.

```python
@keras.utils.register_keras_serializable(package="Product")
class MaskedMeanPooling(layers.Layer):
    """Mean pooling respecting sensor mask."""
    
    def call(self, inputs):
        features, mask = inputs  # (B, S, D), (B, S)
        
        mask_exp = keras.ops.expand_dims(mask, -1)
        masked_sum = keras.ops.sum(features * mask_exp, axis=1)
        denom = keras.ops.sum(mask_exp, axis=1) + 1e-8
        
        return masked_sum / denom
```

**Decisão:** Usar ambos (atenção + média) e concatenar, dando ao modelo duas perspetivas complementares.

---

## 7. Pipeline de Treino

### 7.1 Configuração Completa

```python
@dataclass
class ProductConfig:
    # === DADOS ===
    max_sensors: int = 8
    seq_len: int = 64           # Janela temporal
    hop: int = 8                # Hop entre janelas
    n_fft: int = 32             # FFT para espectrograma
    n_mels: int = 16            # Mel bins
    
    # === MODELO ===
    cnn_filters: List[int] = [32, 64, 128]
    dense_units: int = 128
    dropout: float = 0.3
    l2_reg: float = 1e-5
    sensor_dropout_rate: float = 0.15
    
    # === TREINO ===
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    lr_decay_factor: float = 0.5
    lr_patience: int = 8
    patience: int = 15          # Early stopping
    
    # === LOSS WEIGHTS ===
    loss_weight_severity: float = 3.0   # Mais importante
    loss_weight_mode: float = 2.5
    loss_weight_rul: float = 1.0
    loss_weight_health: float = 1.5
```

### 7.2 Funções de Loss

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cfg.lr),
    loss={
        "severity": keras.losses.SparseCategoricalCrossentropy(),
        "mode": keras.losses.SparseCategoricalCrossentropy(),
        "rul": keras.losses.Huber(delta=0.1),  # Robusto a outliers
        "health": keras.losses.MeanSquaredError(),
    },
    loss_weights={
        "severity": 3.0,
        "mode": 2.5,
        "rul": 1.0,
        "health": 1.5,
    }
)
```

**Decisões:**
- **SparseCategoricalCrossentropy** para classificação (labels são inteiros)
- **Huber Loss** para RUL (mais robusta a outliers que MSE)
- **MSE** para Health (target contínuo e bem distribuído)

### 7.3 Callbacks

```python
callbacks = [
    # Early Stopping monitoriza accuracy de severity
    keras.callbacks.EarlyStopping(
        monitor='val_severity_accuracy',
        mode='max',
        patience=15,
        restore_best_weights=True,
    ),
    
    # Reduzir LR quando loss plateau
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    ),
    
    # Guardar melhor modelo
    keras.callbacks.ModelCheckpoint(
        'best_cnn_2d.weights.h5',
        monitor='val_severity_accuracy',
        mode='max',
        save_best_only=True,
    ),
]
```

### 7.4 Baseline sklearn (Obrigatório)

Antes de treinar o CNN, é **obrigatório** validar o dataset com modelos sklearn simples:

```python
def run_sklearn_baseline(pipeline, train_df, test_df):
    """FASE 2: Baseline sklearn obrigatório."""
    
    # Severity: RandomForest
    rf_sev = RandomForestClassifier(n_estimators=100)
    rf_sev.fit(X_train, sev_train)
    sev_acc = np.mean(rf_sev.predict(X_test) == sev_test)
    
    # Mode: RandomForest
    rf_mode = RandomForestClassifier(n_estimators=100)
    rf_mode.fit(X_train, mode_train)
    mode_acc = np.mean(rf_mode.predict(X_test) == mode_test)
    
    # RUL: Ridge Regression
    ridge_rul = Ridge(alpha=1.0)
    ridge_rul.fit(X_train, rul_train)
    rul_mae = np.mean(np.abs(ridge_rul.predict(X_test) - rul_test))
    
    # Health: Ridge Regression
    ridge_health = Ridge(alpha=1.0)
    ridge_health.fit(X_train, health_train)
    health_mae = np.mean(np.abs(ridge_health.predict(X_test) - health_test))
    
    # Se baseline fraco, dados têm problemas
    baseline_ok = sev_acc > 0.85 and mode_acc > 0.85
```

**Resultados do Baseline:**
- Severity: **97.91%**
- Mode: **100%**
- RUL MAE: **4.53%**
- Health MAE: **3.43%**

O baseline forte confirma que o dataset está bem construído.

---

## 8. Decisões de Design

### 8.1 Porquê CNN 2D em vez de 1D?

| Aspeto | CNN 1D | CNN 2D |
|--------|--------|--------|
| Input | Série temporal | Espectrograma |
| Padrões | Temporais | Tempo-frequência |
| Vibração | Bom | **Excelente** |
| Complexidade | Menor | Maior |

**Decisão:** Para análise de vibração industrial, os padrões frequenciais são críticos para identificar modos de falha específicos (ex: cavitação tem assinatura frequencial distinta).

### 8.2 Porquê Sensor Mask?

**Problema:** Em produção, sensores podem falhar ou ser removidos temporariamente.

**Solução:** O modelo aceita um número variável de sensores (1-8) através de uma máscara binária.

```python
# Exemplo: bomba com apenas 5 sensores ativos
sensor_mask = [1, 1, 1, 1, 1, 0, 0, 0]
```

### 8.3 Porquê Multi-Task Learning?

Treinar os 4 outputs simultaneamente oferece vantagens:

1. **Shared representations:** Features aprendidas para severity ajudam mode
2. **Regularização implícita:** Múltiplas tarefas previnem overfitting
3. **Eficiência:** Um único modelo em vez de 4

### 8.4 Porquê Loss Weights Diferentes?

```python
loss_weights = {
    "severity": 3.0,  # ← Mais importante
    "mode": 2.5,
    "rul": 1.0,
    "health": 1.5,
}
```

**Justificação:** A classificação de severity é a tarefa mais crítica para decisões de manutenção. Se o modelo classificar incorretamente um estado "failure" como "normal", as consequências são graves.

### 8.5 Porquê Early Stopping em Severity Accuracy?

```python
EarlyStopping(monitor='val_severity_accuracy', mode='max')
```

O modelo pára quando a accuracy de severity deixa de melhorar, priorizando esta métrica sobre as outras.

---

## 9. Resultados Obtidos

### 9.1 Métricas Finais

| Métrica | Resultado | Target | Status |
|---------|-----------|--------|--------|
| **Severity Accuracy** | 93.44% | > 90% | ✅ PASS |
| **Mode Accuracy** | 100.00% | > 90% | ✅ PASS |
| **RUL MAE** | 3.06% | < 20% | ✅ PASS |
| **Health MAE** | 2.96% | < 10% | ✅ PASS |

**Targets cumpridos: 4/4** ✅

### 9.2 Classification Report - Severity

```
              precision    recall  f1-score   support

       early       0.92      0.93      0.93       500
     failure       0.94      0.96      0.95       352
    moderate       0.91      0.96      0.93       554
      normal       0.96      0.95      0.96       605
      severe       0.94      0.87      0.90       475

    accuracy                           0.93      2486
   macro avg       0.93      0.93      0.93      2486
```

### 9.3 Classification Report - Mode

```
                  precision    recall  f1-score   support

    bearing_wear       1.00      1.00      1.00       817
      cavitation       1.00      1.00      1.00       808
       imbalance       1.00      1.00      1.00       102
    misalignment       1.00      1.00      1.00       116
normal_operation       1.00      1.00      1.00       643

        accuracy                           1.00      2486
```

### 9.4 Matriz de Confusão - Severity

```
            early  failure  moderate  normal  severe
early         464        0        13      23       0
failure         0      339         0       0      13
moderate       11        0       532       0      11
normal         28        0         0     577       0
severe          0       22        42       0     411
```

**Observações:**
- Melhor performance: **failure** (96% recall) - crítico para segurança
- Alguma confusão entre **severe** e **moderate** (esperado pela proximidade)
- **Normal** bem separado das outras classes

### 9.5 Curva de Aprendizagem

O modelo convergiu após **19 epochs** (early stopping aos 34):

- Época 1: severity_acc = 0.75
- Época 10: severity_acc = 0.89
- Época 19: severity_acc = 0.93 (melhor)
- Época 34: early stopping

---

## 10. Validação e Testes de Stress

### 10.1 Teste de Robustez a Sensores em Falta

**Cenário:** 30% dos sensores são desligados aleatoriamente.

```python
# Simular falha de sensores
n_drop = int(max_sensors * 0.30)  # 30% = 2-3 sensores
for i in range(len(masks)):
    drop_idx = np.random.choice(max_sensors, n_drop, replace=False)
    masks[i, drop_idx] = 0
```

**Resultado:** Severity Accuracy = **92.16%** (apenas -1.28% degradação)

### 10.2 Teste de Robustez a Ruído

**Cenário:** Adicionar ruído gaussiano (σ=0.1) aos espectrogramas.

```python
specs_noisy = specs + np.random.normal(0, 0.1, specs.shape)
```

**Resultado:** Severity Accuracy = **93.08%** (apenas -0.36% degradação)

### 10.3 Sumário de Robustez

| Teste | Accuracy Original | Accuracy com Stress | Degradação |
|-------|-------------------|---------------------|------------|
| Normal | 93.44% | 93.44% | - |
| 30% sensores em falta | 93.44% | 92.16% | -1.28% |
| Ruído gaussiano | 93.44% | 93.08% | -0.36% |

**Veredicto: ROBUSTO** ✅

---

## 11. Conclusões

### 11.1 Objetivos Cumpridos

✅ Modelo CNN 2D desenvolvido e validado  
✅ Todos os 4 targets de performance cumpridos  
✅ Testes de robustez passados  
✅ Código organizado e documentado  
✅ Pipeline reprodutível com seed fixo  

### 11.2 Contribuições Técnicas

1. **Arquitetura Multi-Sensor com Máscara:** Permite operação com número variável de sensores
2. **SensorDropout Layer:** Aumenta robustez a falhas de sensores
3. **Dataset v2:** Metodologia para gerar dados sintéticos com separabilidade garantida
4. **Pipeline Completo:** Desde geração de dados até deployment

### 11.3 Limitações

1. **Dados sintéticos:** O modelo foi treinado em dados simulados. Validação com dados reais é necessária.
2. **Domínio específico:** Otimizado para bombas industriais. Transferência para outros equipamentos requer fine-tuning.
3. **Sem RNN/LSTM:** A arquitetura atual não captura dependências de longo prazo.

### 11.4 Trabalho Futuro

1. **Validação com dados reais** de bombas industriais
2. **Transfer Learning** para outros tipos de equipamentos
3. **Integração com Digital Twin** em tempo real
4. **Explicabilidade (XAI)** - SHAP values para interpretar decisões

### 11.5 Modelo Aprovado para Produção

```
======================================================================
FINAL SUMMARY
======================================================================
  TARGETS MET: 4/4
  Robustness:  PASS
  Baseline:    PASS

  ✅ MODEL APPROVED FOR PRODUCTION
======================================================================
```

---

## 12. Referências Técnicas

### 12.1 Frameworks e Bibliotecas

- TensorFlow 2.x / Keras 3.x - https://www.tensorflow.org/
- scikit-learn - https://scikit-learn.org/
- NumPy - https://numpy.org/
- Pandas - https://pandas.pydata.org/

### 12.2 Conceitos de Deep Learning

- Convolutional Neural Networks (CNN) - LeCun et al., 1998
- Attention Mechanisms - Vaswani et al., 2017 ("Attention Is All You Need")
- Multi-Task Learning - Caruana, 1997

### 12.3 Manutenção Preditiva

- Remaining Useful Life (RUL) estimation - Si et al., 2011
- Condition-Based Maintenance - Jardine et al., 2006

### 12.4 Ficheiros do Projeto

| Ficheiro | Descrição |
|----------|-----------|
| `training/train_cnn_2d.py` | Pipeline principal CNN 2D |
| `training/train_cnn_simple.py` | Pipeline CNN 1D simplificado |
| `generators/generate_dataset_v2.py` | Gerador de dataset |
| `models/pump_cnn_2d_product.keras` | Modelo treinado |
| `outputs/reports/eval_report_cnn_2d.json` | Relatório de avaliação |

---

**Fim do Relatório**

*Documento gerado automaticamente em Janeiro 2026*
