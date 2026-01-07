# Bloqueio: Deeploy Snitch - Operador Reshape Não Suportado

**Data:** 25 de Dezembro de 2025  
**Status:** ❌ **BLOQUEADO** - Deeploy Snitch não suporta Reshape do modelo

---

## 🎯 Objetivo

Converter `pump_predictive.onnx` para código C otimizado usando Deeploy targeting Snitch cluster RISC-V.

---

## ✅ Progresso Completado

### 1. Docker Setup
- ✅ Imagem Docker: `ghcr.io/pulp-platform/deeploy:main`
- ✅ Volumes configurados
- ✅ Scripts automáticos criados

### 2. ONNX Correções
- ✅ **Problema 1:** Batch dimensions dinâmicas (`unk__64`, `unk__65`, etc.)
  - **Solução:** Fixar batch=1 em todos inputs/outputs
  - **Arquivo:** `pump_predictive_fixed.onnx`
  
```python
# Fix aplicado
for inp in model.graph.input:
    inp.type.tensor_type.shape.dim[0].dim_value = 1
    inp.type.tensor_type.shape.dim[0].ClearField("dim_param")
```

### 3. Tentativas Executadas
```bash
# Tentativa 1: Batch dinâmico
❌ numpy.core._exceptions._UFuncNoLoopError: ufunc 'multiply' did not contain a loop

# Tentativa 2: Batch fixo
❌ RuntimeError: No mapping found for node [...] with op type Reshape
```

---

## ❌ Bloqueador Crítico

### Erro Final
```
RuntimeError: No mapping found for node 
StatefulPartitionedCallpump_predictive_model_1bn1_1_1batchnormmul_1__12 
with op type Reshape
```

### Análise do Modelo

**Estatísticas do ONNX:**
- Total nodes: 68
- Operações: Conv(6), MatMul(12), Add(15), Relu(15), MaxPool(2), Softmax(2), **Reshape(1)**, etc.

**Node problemático:**
```
Type: Reshape
Name: StatefulPartitionedCall/pump_predictive_model_1/bn1_1_1/batchnorm/mul_1__12
Localização: Dentro de BatchNormalization layer
```

### Causa Raiz

**Deeploy Snitch target não implementa mapeamento para operador Reshape.**

Segundo a arquitetura do Deeploy:
1. Cada `Platform` (Snitch, Chimera, CortexM, etc.) define seu próprio conjunto de operadores suportados
2. Operadores são mapeados para kernels específicos (PULP_NN, CMSIS-NN, etc.)
3. Se não há mapeamento → RuntimeError durante `_bindLayers()`

**Verificação:**
```python
# Em Deeploy/DeeployTypes.py linha 2569
def _selectEngine(self, node):
    # ... tenta encontrar engine para node.op_type
    if not found:
        raise RuntimeError(f"No mapping found for node {node.name} with op type {node.op}")
```

---

## 🔍 Investigação: Operadores Suportados

### Deeploy Snitch Operators (baseado em docs)

**Definitivamente suportados:**
- Convolution (Conv2D)
- MatMul / Gemm
- Add, Sub, Mul
- ReLU, MaxPool
- BatchNormalization (teoricamente)

**Provavelmente NÃO suportados:**
- ❌ Reshape (confirmado pelo erro)
- ❓ Squeeze
- ❓ GlobalAveragePool
- ❓ Softmax em shapes específicos
- ❓ ReduceMean em dimensões arbitrárias

### Por que BatchNorm gerou Reshape?

Durante conversão Keras→SavedModel→ONNX:
1. Keras BatchNormalization é expandida em operações elementares
2. tf2onnx tenta otimizar/fundir operações
3. Algumas operações intermediárias viram Reshape para broadcasting
4. Este Reshape específico (`batchnorm/mul_1`) é necessário mas não suportado

---

## 🛠️ Possíveis Soluções

### Opção A: Simplificar Modelo (Remover/Fundir BatchNorm)
**Abordagem:**
1. Re-treinar modelo SEM BatchNormalization
2. Ou fundir BN nos pesos Conv (foldable)
3. Re-exportar para ONNX
4. Tentar Deeploy novamente

**Prós:** Pode funcionar  
**Contras:** Perda de precisão, re-training necessário  
**Tempo:** 2-4 horas

### Opção B: Implementar Suporte Reshape em Deeploy
**Abordagem:**
1. Criar `ReshapeEngine` para Snitch
2. Adicionar mapeamento em `Platforms/Snitch.py`
3. Implementar kernel RISC-V
4. Rebuild Deeploy

**Prós:** Solução completa  
**Contras:** Complexo, requer conhecimento profundo do Deeploy  
**Tempo:** 1-2 semanas

### Opção C: Usar Platform Diferente (Chimera/CortexM)
**Abordagem:**
1. Tentar `testRunner_chimera.py` ou `testRunner_cortexm.py`
2. Verificar se suportam mais operadores
3. Adaptar código gerado para Snitch

**Prós:** Pode ter mais operadores  
**Contras:** Código não-otimizado para Snitch  
**Tempo:** 1-2 horas

### Opção D: Usar Alternativa ao Deeploy
**Abordagem:**
1. **TVM** (Apache TVM) - suporta RISC-V + ONNX
2. **CMSIS-NN** manual - escrever código C manualmente
3. **TFLite Micro** - converter para TFLite → C++
4. **Glow** (Facebook) - ONNX → C backend

**Prós:** Ferramentas maduras com mais suporte  
**Contras:** Podem não ter otimizações Snitch-specific  
**Tempo:** Variável (2-5 dias)

---

## 📊 Estado Atual dos Arquivos

### Modelos ONNX Criados
```
~/onnx_export/
├── pump_predictive.onnx         # Original (batch dinâmico)
├── pump_predictive_fixed.onnx   # Batch=1 fixo (ainda falha)
└── pump_test/network.onnx       # Cópia do fixed
```

### Scripts Docker
```
~/onnx_export/
├── run_deeploy_docker.sh        # Container interativo
├── deeploy_workflow.sh          # Workflow automático
└── deeploy_run.log              # Log completo da tentativa
```

### Deeploy Output (parcial)
```
~/onnx_export/deeploy_repo/DeeployTest/
└── TEST_SNITCH/pump_test/       # Pasta criada mas vazia (geração falhou)
```

---

## 🎯 Recomendação Final

### **Curto Prazo (Hoje/Amanhã):**

**Opção A (Simplificar Modelo)** - Mais pragmática:
1. Exportar modelo sem BatchNorm OU fundir BN nos pesos Conv
2. Re-exportar ONNX com apenas operadores básicos
3. Verificar com `onnx.checker`
4. Tentar Deeploy novamente

**Comandos:**
```python
# Fundir BatchNorm nos pesos Conv
from onnxconverter_common import optimizer
optimized = optimizer.optimize_onnx(model, 
    optimization_options=['fuse_bn_into_conv'])
```

### **Médio Prazo (Esta Semana):**

**Opção D (TVM)** - Mais robusto:
1. Instalar Apache TVM com RISC-V backend
2. Converter ONNX → Relay IR → C/ASM
3. Compilar para Snitch com optimizações
4. Testar no simulador

**Documentação:** https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html

---

## 🔗 Referências Úteis

- **Deeploy Supported Ops:** https://pulp-platform.github.io/Deeploy/features.html
- **ONNX Optimizer:** https://github.com/onnx/optimizer
- **TVM RISC-V:** https://tvm.apache.org/docs/how_to/deploy_models/deploy_model_on_riscv.html
- **TFLite Micro:** https://www.tensorflow.org/lite/microcontrollers
- **PULP_NN Kernels:** https://github.com/pulp-platform/pulp-nn

---

## ✅ Decisão Necessária

**Qual caminho seguir?**
- [ ] A - Simplificar modelo (remover BN)
- [ ] B - Implementar Reshape no Deeploy
- [ ] C - Tentar platform diferente
- [ ] D - Usar TVM ou outra ferramenta

**Favor decidir para continuar o workflow até o Snitch cluster.**
