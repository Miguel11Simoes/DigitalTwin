# Guia: Deeploy Docker - ONNX → C Code Generation

**Data:** 25 de Dezembro de 2025  
**Status:** Em execução ✅

---

## 🎯 Objetivo

Converter `pump_predictive.onnx` (1.8 MB) para código C otimizado usando Deeploy dentro de Docker, eliminando problemas de toolchains locais.

---

## 📦 Setup Completo

### 1. Imagem Docker
```bash
docker pull ghcr.io/pulp-platform/deeploy:main
```

**Tamanho:** ~2-3 GB  
**Conteúdo:** LLVM RISC-V, GCC cross-compiler, Snitch simulators, Deeploy

### 2. Estrutura de Volumes

```
Host (~/onnx_export)          →  Container (/workspace)
├── pump_test/                →  /workspace/pump_test/
│   ├── network.onnx          →  Modelo ONNX (1.8 MB)
│   ├── inputs.npz            →  Test inputs (aux: 1×102, spec: 1×128×128×1)
│   └── outputs.npz           →  Expected outputs (4 tensors)
└── deeploy_repo/             →  /app/Deeploy/ (código gerado persistido)
```

### 3. Scripts Criados

**`run_deeploy_docker.sh`** (interativo):
```bash
#!/bin/bash
docker run -it --rm --name deeploy_main \
  -v ~/onnx_export:/workspace \
  -v ~/onnx_export/deeploy_repo:/app/Deeploy \
  ghcr.io/pulp-platform/deeploy:main
```

**`deeploy_workflow.sh`** (automático):
```bash
#!/bin/bash
# 1. Instala Deeploy com pip
# 2. Verifica estrutura pump_test/
# 3. Mostra opções do testRunner
# 4. Gera código C: testRunner_snitch.py -t /workspace/pump_test --cores 8
```

---

## ⚙️ Comando de Execução

```bash
docker run --rm \
  -v ~/onnx_export:/workspace \
  -v ~/onnx_export/deeploy_repo:/app/Deeploy \
  ghcr.io/pulp-platform/deeploy:main \
  bash /workspace/deeploy_workflow.sh
```

---

## 🔧 Correção Crítica: Flags do testRunner

### ❌ ERRADO (o que eu fiz antes):
```bash
python testRunner_snitch.py -t pump_test --skipgen --skipsim
```
**Problema:** `--skipgen` **PULA A GERAÇÃO** (exatamente o oposto do que queremos!)

### ✅ CORRETO:
```bash
# Opção 1: Gerar + Compilar + Simular (completo)
python testRunner_snitch.py -t /workspace/pump_test --cores 8

# Opção 2: Gerar + Compilar (sem simulação)
python testRunner_snitch.py -t /workspace/pump_test --cores 8 --skipsim

# Opção 3: Só gerar (sem compilar nem simular) - USAR ISTO SE TOOLCHAIN DER ERRO
python testRunner_snitch.py -t /workspace/pump_test --cores 8 --skipsim --skipcompile
```

**Flags disponíveis:**
- `--skipgen`: ❌ Pula geração de código (NÃO USAR!)
- `--skipsim`: ✅ Pula simulação Snitch
- `--skipcompile`: ✅ Pula compilação com GCC/LLVM (não existe flag oficial, verificar `--help`)
- `--toolchain <LLVM|GCC>`: Escolhe toolchain (default: LLVM)
- `--cores <N>`: Número de cores Snitch (default: 9, usando 8)

---

## 📁 Output Esperado

### Estrutura gerada:
```
/app/Deeploy/DeeployTest/TEST_pump_test/  (dentro do container)
├── src/
│   ├── network.c          # Código C principal
│   ├── network.h          # Headers
│   ├── weights.c          # Pesos do modelo
│   └── layer_*.c          # Kernels PULP_NN otimizados
├── build/
│   ├── Makefile           # Build system RISC-V
│   └── CMakeLists.txt
└── deploy/
    └── network.elf        # Binary compilado (se não --skipsim)
```

### Como extrair para o host:
```bash
# Dentro do container
cp -r /app/Deeploy/DeeployTest/TEST_* /workspace/

# No host
ls ~/onnx_export/TEST_pump_test/
```

---

## 🧪 Validação

### Checks pós-geração:
1. **Código C gerado:** `ls TEST_pump_test/src/network.c`
2. **Tamanho dos pesos:** `du -h TEST_pump_test/src/weights.c` (deve ser ~1.8 MB)
3. **Kernels PULP_NN:** `grep -r "PULP_NN" TEST_pump_test/src/`
4. **Arquitetura RISC-V:** `grep -r "riscv" TEST_pump_test/build/`

### Inputs processados:
```python
# O testRunner converte inputs.npz automaticamente
aux: (1, 102) float32 → int8_t/uint8_t (quantizado)
spec: (1, 128, 128, 1) float32 → int8_t (quantizado)
```

---

## 🐞 Troubleshooting

### Erro: "LLVM_INSTALL_DIR is not set"
**Causa:** Toolchain não disponível (mesmo dentro do Docker)  
**Solução:** Verificar que Docker image é `ghcr.io/pulp-platform/deeploy:main` (não `:latest`)

### Erro: "ONNX node not supported"
**Causa:** Modelo usa operadores não implementados no Deeploy  
**Solução:** Verificar `network.onnx` com `onnx.checker`, simplificar modelo

### Warning: "protobuf version mismatch"
**Causa:** Conflitos de versão (normal)  
**Solução:** Ignorar (Docker já tem versões corretas)

---

## 📊 Estatísticas Esperadas

**Input:**
- Modelo ONNX: 1.8 MB
- Inputs: 65 KB (aux: 408B, spec: 64KB)
- Outputs: 88 bytes (4 tensors)

**Output esperado:**
- network.c: ~50-200 KB (depende de otimizações)
- weights.c: ~1.8 MB (pesos quantizados)
- network.elf: ~2-3 MB (se compilado)

**Target:**
- Plataforma: Snitch cluster
- Cores: 8
- ISA: RISC-V RV32IMC
- Backend: PULP_NN v3

---

## 🔗 Referências

- **Deeploy Docs:** https://pulp-platform.github.io/Deeploy/
- **testRunner Source:** `/app/Deeploy/DeeployTest/testRunner_snitch.py`
- **Docker Image:** ghcr.io/pulp-platform/deeploy:main
- **Snitch Docs:** https://pulp-platform.github.io/snitch/

---

## ✅ Progresso

- [x] Docker image downloaded
- [x] Volumes configurados
- [x] Scripts criados
- [x] Container executado
- [ ] Código C gerado (em andamento...)
- [ ] Validação do output

---

**Próximos passos:** Esperar conclusão da geração → Validar código C → Copiar para backend/models/
