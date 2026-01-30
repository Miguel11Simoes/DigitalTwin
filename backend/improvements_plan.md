# Plano de Melhorias para Atingir Targets (5 épocas)

## Targets ao Epoch 5:
- ✅ val_severity_acc ≥ 0.85
- ✅ val_mode_acc ≥ 0.80
- ✅ val_mode_top2 ≥ 0.92
- ✅ val_health_mae a cair de forma estável
- ✅ val_rul_mae ≤ 0.20

## Problemas Identificados (Treino Anterior):
- val_mode_acc: 0.36-0.70 ❌ (Target: 0.80)
- val_severity_acc: 0.26-0.52 ❌ (Target: 0.85)
- val_mode_top2: 0.72-0.82 ❌ (Target: 0.92)

## Melhorias a Implementar (Ordem de Impacto):

### 1. FOCAL LOSS (Maior Impacto - Classes Raras)
**Problema:** Cross-entropy ignora classes raras, modelo aprende apenas dominantes
**Solução:** Focal Loss com gamma=2.0, alpha=0.25

### 2. AUMENTAR REGULARIZAÇÃO (Fix Overfitting)
**Problema:** Gap grande train vs val
**Solução:**
- Dropout: 0.30 → 0.50
- L2: 1e-4 → 1e-3
- BatchNormalization em todas Dense layers

### 3. CLASS BALANCING MELHORADO
**Problema:** sample_weight não suficiente
**Solução:**
- Balanced batching (cada batch com distribuição uniforme)
- Oversampling de classes raras (SMOTE ou duplicate)

### 4. LEARNING RATE SCHEDULE
**Problema:** LR constante pode não convergir bem
**Solução:**
- Warmup: 0 → 3e-4 em 3 épocas
- Cosine Annealing: 3e-4 → 1e-5 em 120 épocas

### 5. LABEL SMOOTHING REMOVAL
**Problema:** Label smoothing prejudica classes raras
**Solução:** Remover label_smoothing se existir

### 6. AUMENTAR BATCH SIZE
**Problema:** batch_size=32 dá gradientes ruidosos
**Solução:** batch_size=64 (mais estável)

## Implementação Prioritária (Se < Target):

**ITERAÇÃO 1 (Crítico):**
1. Focal Loss para mode e severity
2. Dropout 0.50
3. Batch size 64

**ITERAÇÃO 2 (Se ainda não atingir):**
4. BatchNormalization em todas Dense
5. Balanced batching
6. L2 1e-3

**ITERAÇÃO 3 (Fine-tuning):**
7. Warmup + Cosine Annealing
8. SMOTE oversampling
9. Remover label smoothing
