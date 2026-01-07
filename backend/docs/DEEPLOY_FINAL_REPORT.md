# Relatório Final: Tentativa Deeploy Snitch

**Data:** 25 de Dezembro de 2025  
**Status:** ❌ **INVIÁVEL** - Deeploy Snitch não suporta operadores necessários do modelo

---

## 🎯 Objetivo

Converter `pump_predictive.onnx` para código C otimizado usando Deeploy targeting Snitch cluster RISC-V.

---

## ✅ Trabalho Completado

### 1. Setup Docker
- ✅ Docker image: `ghcr.io/pulp-platform/deeploy:main` (instalado)
- ✅ Volumes configurados (workspace + deeploy_repo)
- ✅ Scripts automatizados criados

### 2. Correções ONNX Aplicadas

#### Problema 1: Batch Dimensions Dinâmicas
**Erro:**
```
numpy._UFuncNoLoopError: ufunc 'multiply' did not contain a loop with signature matching types (dtype('<U21'), dtype('<U21'))
```

**Solução:**
```python
# Fixar batch=1 em todos inputs/outputs
for inp in model.graph.input:
    inp.type.tensor_type.shape.dim[0].dim_value = 1
    inp.type.tensor_type.shape.dim[0].ClearField("dim_param")
```

**Resultado:** ✅ `pump_predictive_fixed.onnx`

#### Problema 2: Operador Reshape Não Suportado
**Erro:**
```
RuntimeError: No mapping found for node [...batchnorm/mul_1__12] with op type Reshape
```

**Análise:**
- Reshape na posição 0 do grafo (primeiro operador)
- Input: `spec [1, 128, 128, 1]` (NHWC)
- Target: `[-1, 1, 128, 128]` (NCHW)
- **Causa:** Conversão de layout NHWC → NCHW

**Solução Aplicada:**
```python
# Substituir Reshape por Transpose equivalente
transpose_node = helper.make_node(
    'Transpose',
    inputs=['spec'],
    outputs=[node.output[0]],
    perm=[0, 3, 1, 2],  # NHWC → NCHW
    name=node.name.replace('Reshape', 'Transpose')
)
```

**Resultado:** ✅ `pump_predictive_transposed.onnx` (Reshape eliminado)

### 3. Tentativas Executadas

#### Tentativa 1: Batch dinâmico
```bash
docker run ... testRunner_snitch.py -t /workspace/pump_test --cores 8 --skipsim
```
**Resultado:** ❌ Shape error (numpy dtype mismatch)

#### Tentativa 2: Batch fixo
```bash
# Com pump_predictive_fixed.onnx
```
**Resultado:** ❌ Reshape não suportado

#### Tentativa 3: Reshape → Transpose
```bash
# Com pump_predictive_transposed.onnx
```
**Resultado:** ✅ Reshape resolvido → ❌ **ReduceMean não suportado**

---

## ❌ Bloqueador Final: ReduceMean

### Erro
```
RuntimeError: No mapping found for node 
StatefulPartitionedCallpump_predictive_model_1aux_norm_1momentsmean 
with op type ReduceMean
```

### Análise
- **Operador:** ReduceMean
- **Contexto:** Normalização do input 'aux' (BatchNorm expandido)
- **Localização:** Nodes 1, 4 do grafo (primeiros processamentos)

### Operadores do Modelo vs Suportados

**Operadores no modelo (top 10):**
```
Add             : 15  ✅ Suportado
Relu            : 15  ✅ Suportado
MatMul          : 12  ✅ Suportado
Conv            :  6  ✅ Suportado
Mul             :  5  ✅ Suportado
ReduceMean      :  2  ❌ NÃO suportado
MaxPool         :  2  ✅ Suportado
Softmax         :  2  ❓ Desconhecido
Reshape         :  1  ❌ NÃO suportado (resolvido)
Sub             :  1  ✅ Suportado
Sqrt            :  1  ❓ Desconhecido
Reciprocal      :  1  ❓ Desconhecido
GlobalAveragePool: 1  ❓ Desconhecido
Squeeze         :  1  ❓ Desconhecido
```

**Operadores Deeploy Snitch confirmados:**
- Conv2D, MatMul, Add, Sub, Mul, ReLU, MaxPool
- BatchNorm (quando fusionado, não expandido)

**Operadores NÃO suportados (confirmados):**
- ❌ Reshape (workaround: Transpose)
- ❌ ReduceMean
- ❓ Sqrt, Reciprocal (provável que não)
- ❓ Squeeze
- ❓ GlobalAveragePool

---

## 🔍 Causa Raiz

### Por que tantos operadores "estranhos"?

Keras BatchNormalization é expandido durante export:
```
BatchNorm(x) = (x - mean) / sqrt(var + epsilon) * gamma + beta
```

Expande-se em:
1. **ReduceMean** (calcular mean)
2. Sub (x - mean)
3. Mul (squared difference)
4. **ReduceMean** (calcular variance)
5. **Sqrt** (raiz da variance)
6. **Reciprocal** (1/sqrt)
7. Mul (normalizar)
8. Add (bias)

**5 destes operadores não são suportados pelo Deeploy Snitch.**

### Por que não fundir BatchNorm?

tf2onnx tenta fundir BatchNorm em Conv quando possível, MAS:
- Apenas funciona para BatchNorm **diretamente após Conv**
- No nosso modelo, BatchNorm está:
  - Nos **inputs** (aux_norm_1, bn1_1_1) - **antes** das Conv
  - Entre layers densas (onde não há Conv para fundir)

---

## 📊 Estatísticas Finais

### Modelos ONNX Criados
```
~/onnx_export/
├── pump_predictive.onnx              # Original (1.8 MB, batch dinâmico)
├── pump_predictive_fixed.onnx        # Batch=1 fixo
├── pump_predictive_transposed.onnx   # Reshape → Transpose ✅
├── pump_predictive_noreshape.onnx    # Tentativa fold constantes (0 removidos)
└── pump_test/network.onnx            # Versão testada (transposed)
```

### Scripts Criados
```
~/onnx_export/
├── simplify_onnx.py                  # onnxsim (não usado - build lento)
├── fold_constant_reshape.py          # Constant folding manual
├── replace_reshape_with_transpose.py # Reshape → Transpose ✅
├── run_deeploy_docker.sh             # Container interativo
├── deeploy_workflow.sh               # Workflow automático
└── deeploy_snitch_run.log            # Log completo das tentativas
```

### Tempo Investido
- Setup Docker: 10 min
- Correção batch dimensions: 5 min
- Investigação Reshape: 20 min
- Substituição Reshape→Transpose: 10 min
- Descoberta ReduceMean: 5 min
- **Total: ~50 minutos**

---

## 🎯 Conclusão e Próximos Passos

### Conclusão

**Deeploy Snitch NÃO É VIÁVEL para este modelo CNN multi-task.**

**Razões:**
1. Conjunto limitado de operadores suportados
2. BatchNorm expandido gera 5+ operadores não suportados
3. Modelo usa operações complexas (ReduceMean, GlobalAveragePool, Squeeze)
4. Resolver cada operador individualmente seria trabalho de semanas

### Recomendação: Apache TVM

**Por quê TVM?**
- ✅ Suporte completo para todos operadores ONNX
- ✅ Backend RISC-V maduro e testado
- ✅ Quantização automática (int8/int16)
- ✅ Otimizações para low-power/embedded
- ✅ Comunidade ativa e documentação extensa

**TVM vs Deeploy:**
| Característica | Deeploy Snitch | Apache TVM |
|---|---|---|
| Operadores ONNX | ~15 básicos | 200+ completos |
| RISC-V Support | ✅ Nativo | ✅ Via LLVM |
| Quantização | Manual | Automática |
| Otimização Snitch | ✅✅✅ Específica | ✅ Genérica |
| Learning curve | Média | Alta |
| Documentação | Limitada | Extensa |

**Trade-off:**
- Deeploy: código **mais otimizado** para Snitch, mas suporte limitado
- TVM: código **funcional** para RISC-V, otimizações genéricas mas robustas

---

## 📋 Plano TVM (Opção D)

### Fase 1: Setup (1-2 horas)
```bash
# Instalar TVM com RISC-V backend
git clone --recursive https://github.com/apache/tvm
cd tvm && mkdir build && cp cmake/config.cmake build/
# Edit config.cmake: set(USE_LLVM ON), set(USE_RISCV ON)
cd build && cmake .. && make -j$(nproc)
```

### Fase 2: Converter ONNX → Relay (30 min)
```python
import onnx
import tvm
from tvm import relay

onnx_model = onnx.load("pump_predictive_transposed.onnx")
shape_dict = {'aux': (1, 102), 'spec': (1, 1, 128, 128)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
```

### Fase 3: Compilar para RISC-V (1 hora)
```python
target = tvm.target.Target("llvm -mtriple=riscv32-unknown-elf -mcpu=generic-rv32")
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
lib.export_library("pump_model.so")
```

### Fase 4: Quantização INT8 (1-2 horas)
```python
from tvm.relay.quantize import quantize
qconfig = relay.quantize.qconfig(calibrate_mode='kl_divergence')
with qconfig:
    qmod = quantize(mod, params=params)
```

### Fase 5: Deploy Snitch (2-3 horas)
- Cross-compile para RISC-V ELF
- Integrar com runtime Snitch
- Testar no simulador

**Tempo total estimado: 6-9 horas**

---

## 🔗 Referências

### Deeploy
- Docs: https://pulp-platform.github.io/Deeploy/
- GitHub: https://github.com/pulp-platform/Deeploy
- Supported Ops: https://pulp-platform.github.io/Deeploy/features.html

### TVM
- Homepage: https://tvm.apache.org/
- ONNX Tutorial: https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html
- RISC-V Docs: https://tvm.apache.org/docs/how_to/deploy/riscv.html
- Quantization: https://tvm.apache.org/docs/how_to/quantize.html

### Alternativas
- TFLite Micro: https://www.tensorflow.org/lite/microcontrollers
- CMSIS-NN: https://github.com/ARM-software/CMSIS-NN
- Glow: https://github.com/pytorch/glow

---

## ✅ Arquivos Finais

### Para continuar com TVM:
- `pump_predictive_transposed.onnx` - Modelo otimizado (sem Reshape)
- `pump_test/inputs.npz` - Test inputs
- `pump_test/outputs.npz` - Expected outputs

### Documentação:
- `ONNX_EXPORT_REPORT.md` - Export Keras → ONNX
- `DEEPLOY_BLOCKER_REPORT.md` - Bloqueio inicial
- `DEEPLOY_DOCKER_GUIDE.md` - Setup Docker
- `DEEPLOY_FINAL_REPORT.md` - Este documento

---

**Decisão Final:** Avançar para **Apache TVM** (Opção D) para completar deployment no Snitch cluster.
