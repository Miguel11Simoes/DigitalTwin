#!/bin/bash
# Script de Monitoramento Automático do Treino
# Executa 5 épocas e analisa resultados automaticamente

LOGFILE="/tmp/train_5ep_v2.log"
TARGETS_FILE="/tmp/targets_status.txt"

echo "=== MONITORING TRAINING V2 (Focal Loss + Dropout 0.50 + Batch 64) ==="
echo "Target: val_severity_acc ≥ 0.85, val_mode_acc ≥ 0.80, val_mode_top2 ≥ 0.92"
echo ""

# Aguardar treino completar
while ps aux | grep -q "[p]ython3 train_pump_predictive_market.py"; do
    echo "[$(date +'%H:%M:%S')] Treino em execução..."
    sleep 30
done

echo ""
echo "=== TREINO COMPLETADO - EXTRAINDO RESULTADOS ==="
echo ""

# Extrair métricas finais (epoch 5)
LAST_SEV=$(grep "val_severity_acc:" $LOGFILE | tail -n 1 | grep -oP 'val_severity_acc:\s+\K[0-9.]+')
LAST_MODE=$(grep "val_mode_acc:" $LOGFILE | tail -n 1 | grep -oP 'val_mode_acc:\s+\K[0-9.]+')
LAST_TOP2=$(grep "val_mode_top2:" $LOGFILE | tail -n 1 | grep -oP 'val_mode_top2:\s+\K[0-9.]+')

echo "=== RESULTADOS EPOCH 5 ==="
echo "val_severity_acc: $LAST_SEV (target: ≥0.85)"
echo "val_mode_acc: $LAST_MODE (target: ≥0.80)"
echo "val_mode_top2: $LAST_TOP2 (target: ≥0.92)"
echo ""

# Todas as métricas VAL PRODUCT
echo "=== HISTÓRICO COMPLETO ===" 
grep "VAL PRODUCT" $LOGFILE
echo ""

# Verificar targets
{
    echo "=== STATUS DOS TARGETS ==="
    echo "Target val_severity_acc ≥ 0.85: $(awk "BEGIN {print ($LAST_SEV >= 0.85) ? \"✅ PASS\" : \"❌ FAIL ($LAST_SEV)\"}")"
    echo "Target val_mode_acc ≥ 0.80: $(awk "BEGIN {print ($LAST_MODE >= 0.80) ? \"✅ PASS\" : \"❌ FAIL ($LAST_MODE)\"}")"
    echo "Target val_mode_top2 ≥ 0.92: $(awk "BEGIN {print ($LAST_TOP2 >= 0.92) ? \"✅ PASS\" : \"❌ FAIL ($LAST_TOP2)\"}")"
    echo ""
    
    if awk "BEGIN {exit !($LAST_SEV >= 0.85 && $LAST_MODE >= 0.80 && $LAST_TOP2 >= 0.92)}"; then
        echo "✅✅✅ TODOS OS TARGETS ATINGIDOS! ✅✅✅"
        echo "Modelo pronto para treino completo (120 épocas)"
    else
        echo "❌ TARGETS NÃO ATINGIDOS - REQUER ITERAÇÃO 2"
        echo ""
        echo "Próximas melhorias a implementar:"
        echo "1. BatchNormalization em todas Dense layers"
        echo "2. L2 regularization: 1e-4 → 1e-3"
        echo "3. Balanced batching (uniform class distribution per batch)"
        echo "4. SMOTE oversampling para classes raras"
    fi
} | tee $TARGETS_FILE

echo ""
echo "=== RESULTADOS SALVOS EM $TARGETS_FILE ==="
