# Guia Completo: De Modelo Python para MLF (Model Library Format)

Este guia documenta todos os passos necessários para converter um modelo de Deep Learning treinado em Python para o formato MLF (Model Library Format) com código C puro, adequado para deployment em sistemas embarcados como Snitch RISC-V.

## Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Fase 1: Treino do Modelo em Python](#fase-1-treino-do-modelo-em-python)
3. [Fase 2: Exportação para ONNX](#fase-2-exportação-para-onnx)
4. [Fase 3: Instalação e Configuração do TVM](#fase-3-instalação-e-configuração-do-tvm)
5. [Fase 4: Compilação ONNX para MLF](#fase-4-compilação-onnx-para-mlf)
6. [Fase 5: Extração e Uso dos Arquivos](#fase-5-extração-e-uso-dos-arquivos)
7. [Estrutura Final dos Arquivos](#estrutura-final-dos-arquivos)

---

## Pré-requisitos

### Software Necessário
- Python 3.12+ (WSL Ubuntu)
- TensorFlow/Keras 2.x
- ONNX 1.16.0
- TVM 0.15.0
- CMake 3.18+
- LLVM 17
- Git

### Sistema Operacional
- Windows 11 com WSL2 (Ubuntu)
- Ou Linux nativo

---

## Fase 1: Treino do Modelo em Python

### Passo 1.1: Preparar o Script de Treino

**O que fazer:** Criar ou usar um script Python que treina seu modelo com Keras/TensorFlow.

**Arquivo exemplo:** `train_tf.py` ou `train_pump_predictive.py`

**Estrutura básica do modelo:**
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Definir arquitetura do modelo
def create_model(input_shapes, num_classes):
    # Input 1: Features auxiliares
    input_aux = keras.Input(shape=(input_shapes['aux'],), name='aux')
    
    # Input 2: Espectrograma (imagem)
    input_spec = keras.Input(shape=input_shapes['spec'], name='spec')
    
    # Processamento do espectrograma com CNN
    x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_spec)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Flatten()(x)
    
    # Concatenar features
    combined = keras.layers.Concatenate()([input_aux, x])
    
    # Dense layers
    x = keras.layers.Dense(128, activation='relu')(combined)
    x = keras.layers.Dropout(0.5)(x)
    
    # Outputs múltiplos
    output1 = keras.layers.Dense(16, activation='softmax', name='output0')(x)
    output2 = keras.layers.Dense(4, activation='softmax', name='output1')(x)
    output3 = keras.layers.Dense(1, activation='sigmoid', name='output2')(x)
    output4 = keras.layers.Dense(1, activation='linear', name='output3')(x)
    
    model = keras.Model(
        inputs=[input_aux, input_spec],
        outputs=[output1, output2, output3, output4]
    )
    
    return model

# Treinar modelo
model = create_model(
    input_shapes={'aux': 102, 'spec': (128, 128, 1)},
    num_classes={'out1': 16, 'out2': 4}
)

model.compile(
    optimizer='adam',
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy', 'mse'],
    metrics=['accuracy']
)

# model.fit(...)  # Seu código de treino aqui

# Salvar modelo treinado
model.save('models/pump_predictive.keras')
```

### Passo 1.2: Executar o Treino

**Comando:**
```bash
python train_tf.py
```

**Resultado esperado:**
- Arquivo `models/pump_predictive.keras` criado
- Modelo treinado e pronto para exportação

---

## Fase 2: Exportação para ONNX

### Passo 2.1: Instalar Dependências ONNX

**O que fazer:** Instalar bibliotecas necessárias para conversão ONNX.

**Comandos:**
```bash
# No WSL/Ubuntu
pip install tf2onnx onnx onnxruntime
```

### Passo 2.2: Criar Script de Exportação com Transposição

**O que fazer:** A conversão para ONNX requer transposição dos pesos das camadas Conv2D de `channels_last` (formato Keras) para `channels_first` (formato ONNX padrão). Criar script que faz isso automaticamente.

**Arquivo:** `export_to_onnx.py` (ou similar)

**Código completo:**
```python
import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

def transpose_conv2d_weights(model):
    """
    Transpõe pesos Conv2D de channels_last (H,W,C_in,C_out) 
    para channels_first (C_out,C_in,H,W) para compatibilidade ONNX.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if len(weights) > 0:
                # Formato Keras: (H, W, C_in, C_out)
                # Formato ONNX: (C_out, C_in, H, W)
                kernel = weights[0]
                kernel_transposed = np.transpose(kernel, (3, 2, 0, 1))
                
                if len(weights) > 1:  # tem bias
                    layer.set_weights([kernel_transposed, weights[1]])
                else:
                    layer.set_weights([kernel_transposed])
    
    return model

# Carregar modelo Keras
model_path = 'models/pump_predictive.keras'
model = tf.keras.models.load_model(model_path)

print(f"Modelo carregado: {model_path}")
print(f"Inputs: {[inp.name for inp in model.inputs]}")
print(f"Outputs: {[out.name for out in model.outputs]}")

# Transpor pesos Conv2D
model = transpose_conv2d_weights(model)

# Definir especificações dos inputs
input_signature = [
    tf.TensorSpec([1, 102], tf.float32, name='aux'),
    tf.TensorSpec([1, 128, 128, 1], tf.float32, name='spec')
]

# Converter para ONNX
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path='models/pump_predictive_transposed.onnx'
)

print("\n✓ Conversão concluída!")
print(f"Arquivo ONNX salvo em: models/pump_predictive_transposed.onnx")

# Verificar modelo ONNX
onnx_model_check = onnx.load('models/pump_predictive_transposed.onnx')
onnx.checker.check_model(onnx_model_check)
print("✓ Modelo ONNX válido!")
```

### Passo 2.3: Executar Exportação

**Comando:**
```bash
python export_to_onnx.py
```

**Resultado esperado:**
```
Modelo carregado: models/pump_predictive.keras
Inputs: ['aux', 'spec']
Outputs: ['output0', 'output1', 'output2', 'output3']

✓ Conversão concluída!
Arquivo ONNX salvo em: models/pump_predictive_transposed.onnx
✓ Modelo ONNX válido!
```

**Arquivo gerado:**
- `models/pump_predictive_transposed.onnx` (~1.8 MB)

---

## Fase 3: Instalação e Configuração do TVM

### Passo 3.1: Criar Diretório e Clonar TVM

**O que fazer:** Baixar código-fonte do Apache TVM versão 0.15.0.

**Comandos:**
```bash
# No WSL/Ubuntu
cd ~
mkdir tvm_aot
cd tvm_aot

# Clonar repositório TVM
git clone --depth=1 --branch v0.15.0 https://github.com/apache/tvm.git .
```

**Resultado:** Código TVM 0.15.0 baixado em `~/tvm_aot/`

### Passo 3.2: Inicializar Submódulos Essenciais

**O que fazer:** TVM precisa de bibliotecas auxiliares (dlpack, dmlc-core, rang).

**Comandos:**
```bash
cd ~/tvm_aot

# Inicializar apenas submódulos essenciais
git submodule update --init 3rdparty/dlpack
git submodule update --init 3rdparty/dmlc-core
git submodule update --init 3rdparty/rang
```

**Resultado:** Submódulos baixados sem erro.

### Passo 3.3: Criar Ambiente Virtual Python

**O que fazer:** Isolar dependências Python do TVM.

**Comandos:**
```bash
cd ~/tvm_aot

# Criar venv
python3 -m venv .venv

# Ativar venv
source .venv/bin/activate

# Instalar dependências Python
pip install --upgrade pip
pip install "numpy<2.0" scipy decorator attrs typing-extensions psutil tornado cloudpickle
```

**Resultado:** Ambiente virtual criado em `~/tvm_aot/.venv/`

### Passo 3.4: Instalar ONNX no Ambiente TVM

**O que fazer:** Instalar ONNX para importação de modelos.

**Comandos:**
```bash
# Ainda com venv ativado
pip install onnx==1.16.0
```

### Passo 3.5: Instalar Dependências do Sistema (LLVM)

**O que fazer:** TVM precisa de LLVM para compilação e otimização de código. LLVM 17 é compatível com TVM 0.15.0.

**Comandos:**
```bash
# Instalar LLVM 17 e ferramentas de build
sudo apt update
sudo apt install -y llvm-17-dev clang-17 cmake build-essential

# Definir LLVM 17 como padrão
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-17 100
```

**Verificar instalação:**
```bash
llvm-config --version
# Deve mostrar: 17.0.x
```

### Passo 3.6: Configurar Build do TVM

**O que fazer:** Criar arquivo de configuração que habilita LLVM e microTVM.

**Comandos:**
```bash
cd ~/tvm_aot

# Copiar template de configuração
cp cmake/config.cmake .

# Adicionar configurações necessárias
echo 'set(USE_LLVM "llvm-config")' >> config.cmake
echo 'set(USE_LIBBACKTRACE OFF)' >> config.cmake
echo 'set(USE_MICRO ON)' >> config.cmake
```

**Verificar configuração:**
```bash
tail -5 config.cmake
# Deve mostrar as 3 linhas acima
```

### Passo 3.7: Compilar TVM

**O que fazer:** Compilar TVM com suporte a LLVM e microTVM (AOT executor).

**Comandos:**
```bash
cd ~/tvm_aot

# Criar diretório de build
mkdir -p build
cd build

# Configurar com CMake
cmake ..

# Compilar (vai demorar 15-30 minutos)
make -j$(nproc)
```

**Resultado esperado:**
```
[100%] Building CXX object CMakeFiles/tvm_objs.dir/src/target/llvm/...
[100%] Linking CXX shared library libtvm.so
[100%] Built target tvm
```

**Arquivos gerados:**
- `~/tvm_aot/build/libtvm.so` (~57 MB)
- `~/tvm_aot/build/libtvm_runtime.so` (~3 MB)

### Passo 3.8: Configurar Variáveis de Ambiente

**O que fazer:** Criar script para carregar TVM no Python.

**Comandos:**
```bash
cd ~/tvm_aot

# Criar script de ativação
cat > enable_tvm.sh << 'EOF'
#!/bin/bash
export TVM_HOME=~/tvm_aot
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:${PYTHONPATH}
export LD_LIBRARY_PATH=$TVM_HOME/build:${LD_LIBRARY_PATH}
EOF

# Dar permissão de execução
chmod +x enable_tvm.sh
```

### Passo 3.9: Verificar Instalação do TVM com LLVM

**O que fazer:** Confirmar que TVM foi compilado corretamente com suporte LLVM.

**Comandos:**
```bash
# Ativar ambiente
source ~/tvm_aot/.venv/bin/activate
source ~/tvm_aot/enable_tvm.sh

# Testar TVM
python - <<'PY'
import tvm
from tvm._ffi.base import _LIB

print("TVM version:", tvm.__version__)
print("TVM module:", tvm.__file__)
print("Loaded libtvm:", _LIB._name)
print("LLVM enabled:", tvm.runtime.enabled("llvm"))
PY
```

**Resultado esperado:**
```
TVM version: 0.15.0
TVM module: /home/msmiguel/tvm_aot/python/tvm/__init__.py
Loaded libtvm: /home/msmiguel/tvm_aot/build/libtvm.so
LLVM enabled: 1
```

**⚠️ CRÍTICO:** Se `LLVM enabled: 0`, o TVM não foi compilado com LLVM corretamente. Volte ao Passo 3.5 e verifique a instalação do LLVM.

---

## Fase 4: Compilação ONNX para MLF

### Passo 4.1: Preparar Variáveis de Ambiente

**O que fazer:** Carregar ambiente TVM antes da compilação.

**Comandos:**
```bash
cd ~
source ~/tvm_aot/.venv/bin/activate
source ~/tvm_aot/enable_tvm.sh
```

### Passo 4.2: Definir Caminhos dos Arquivos

**O que fazer:** Configurar paths para modelo ONNX de entrada e MLF de saída.

**Comandos:**
```bash
# Path do modelo ONNX (ajuste conforme seu projeto)
MODEL="/mnt/c/Users/msmig/Desktop/Tese/DigitalTwin/backend/models/pump_predictive_transposed.onnx"

# Path de saída (MLF)
OUT="/home/msmiguel/tvm_out/pump_predictive_transposed_mlf.tar"

# Criar diretório de saída
mkdir -p /home/msmiguel/tvm_out
```

### Passo 4.3: Executar Compilação TVMC

**O que fazer:** Usar TVMC (TVM Compiler) para compilar ONNX para MLF com código C puro.

**Comando completo:**
```bash
python -m tvm.driver.tvmc compile "$MODEL" \
    --input-shapes "aux:[1,102],spec:[1,128,128,1]" \
    --target "c -keys=cpu" \
    --runtime crt \
    --runtime-crt-system-lib=1 \
    --executor aot \
    --executor-aot-interface-api=c \
    --executor-aot-unpacked-api=1 \
    --pass-config tir.disable_vectorize=1 \
    --output-format mlf \
    --output "$OUT"
```

**Explicação dos parâmetros:**

| Parâmetro | Descrição |
|-----------|-----------|
| `--input-shapes` | Dimensões dos inputs (DEVE corresponder ao seu modelo) |
| `--target "c -keys=cpu"` | Gerar código C puro para CPU genérica |
| `--runtime crt` | Usar C Runtime (standalone) |
| `--runtime-crt-system-lib=1` | Gerar biblioteca estática |
| `--executor aot` | Ahead-of-Time compilation |
| `--executor-aot-interface-api=c` | API em C (não Python) |
| `--executor-aot-unpacked-api=1` | API desempacotada (mais simples) |
| `--pass-config tir.disable_vectorize=1` | Desabilitar vetorização SIMD |
| `--output-format mlf` | Formato Model Library Format |
| `--output` | Caminho do arquivo .tar de saída |

**Resultado esperado:**
```
WARNING:autotvm:One or more operators have not been tuned...
```
(Este warning é normal - significa que não usamos AutoTVM para otimização)

**Arquivo gerado:**
- `/home/msmiguel/tvm_out/pump_predictive_transposed_mlf.tar` (~8.3 MB)

### Passo 4.4: Verificar Conteúdo do MLF

**O que fazer:** Inspecionar o arquivo TAR gerado.

**Comandos:**
```bash
cd /home/msmiguel/tvm_out

# Listar conteúdo
tar -tf pump_predictive_transposed_mlf.tar | head -20
```

**Conteúdo esperado:**
```
./codegen/host/include/tvmgen_default.h
./codegen/host/src/default_lib0.c
./codegen/host/src/default_lib1.c
./metadata.json
./parameters/default.params
./runtime/
./src/default.relay
```

---

## Fase 5: Extração e Uso dos Arquivos

### Passo 5.1: Extrair MLF

**O que fazer:** Descompactar o arquivo TAR para acessar código C.

**Comandos:**
```bash
cd /home/msmiguel/tvm_out

# Extrair
tar xf pump_predictive_transposed_mlf.tar

# Verificar arquivos C gerados
find . -name "*.c" -o -name "*.h" | grep -E "(default_lib|tvmgen)"
```

**Arquivos principais:**
```
./codegen/host/src/default_lib0.c       # 7.1 MB - implementação do modelo
./codegen/host/src/default_lib1.c       # 128 KB - runtime adicional
./codegen/host/include/tvmgen_default.h # API C para inferência
```

### Passo 5.2: Inspecionar Header da API

**O que fazer:** Ver definições de inputs/outputs.

**Comandos:**
```bash
head -60 codegen/host/include/tvmgen_default.h
```

**Conteúdo esperado:**
```c
#ifndef TVMGEN_DEFAULT_H_
#define TVMGEN_DEFAULT_H_

// Tamanhos dos tensores em bytes
#define TVMGEN_DEFAULT_SPEC_SIZE 65536    // 128×128×1 floats
#define TVMGEN_DEFAULT_AUX_SIZE 408       // 102 floats
#define TVMGEN_DEFAULT_OUTPUT0_SIZE 64    // 16 floats
// ...

// Estrutura de inputs
struct tvmgen_default_inputs {
    void* aux;
    void* spec;
};

// Estrutura de outputs
struct tvmgen_default_outputs {
    void* output0;
    void* output1;
    void* output2;
    void* output3;
};

// Função principal de inferência
int32_t tvmgen_default_run(
    struct tvmgen_default_inputs* inputs,
    struct tvmgen_default_outputs* outputs
);
```

### Passo 5.3: Verificar Metadata

**O que fazer:** Examinar informações do modelo compilado.

**Comandos:**
```bash
python3 -m json.tool metadata.json | head -80
```

**Informações disponíveis:**
- Versão do TVM (0.15.0)
- Tipos de dados (float32)
- Tamanhos de workspace (~4.6 MB)
- Tamanhos de constantes (~1.8 MB)
- Inputs: `aux` (408 bytes), `spec` (65536 bytes)
- Outputs: 4 tensores

### Passo 5.4: Copiar para Projeto Windows

**O que fazer:** Mover arquivos MLF para o diretório do projeto (se em WSL).

**Comandos:**
```bash
# Copiar TAR
cp /home/msmiguel/tvm_out/pump_predictive_transposed_mlf.tar \
   /mnt/c/Users/msmig/Desktop/Tese/DigitalTwin/backend/models/

# Criar diretório e extrair
cd /mnt/c/Users/msmig/Desktop/Tese/DigitalTwin/backend/models
mkdir -p mlf_output
cd mlf_output
tar xf ../pump_predictive_transposed_mlf.tar
```

**Resultado:** Arquivos acessíveis em:
- `C:\Users\msmig\Desktop\Tese\DigitalTwin\backend\models\pump_predictive_transposed_mlf.tar`
- `C:\Users\msmig\Desktop\Tese\DigitalTwin\backend\models\mlf_output\`

### Passo 5.5: Fazer Commit no Git

**O que fazer:** Versionar arquivos MLF no repositório.

**Comandos:**
```bash
cd /mnt/c/Users/msmig/Desktop/Tese/DigitalTwin

git add backend/models/pump_predictive_transposed_mlf.tar
git add backend/models/mlf_output/
git commit -m "Add TVM MLF compiled model for Snitch RISC-V deployment"
```

---

## Estrutura Final dos Arquivos

### Arquivos Python (origem)
```
backend/
├── train_tf.py                          # Script de treino
├── export_to_onnx.py                    # Script de exportação ONNX
└── models/
    ├── pump_predictive.keras            # Modelo treinado (~5 MB)
    └── pump_predictive_transposed.onnx  # Modelo ONNX (~1.8 MB)
```

### Arquivos MLF (gerados)
```
backend/models/
├── pump_predictive_transposed_mlf.tar   # Arquivo completo (8.3 MB)
└── mlf_output/                          # Conteúdo extraído
    ├── codegen/
    │   └── host/
    │       ├── src/
    │       │   ├── default_lib0.c       # Código do modelo (7.1 MB)
    │       │   └── default_lib1.c       # Runtime (128 KB)
    │       └── include/
    │           └── tvmgen_default.h     # API C (1.6 KB)
    ├── runtime/                         # TVM C runtime standalone
    │   ├── CMakeLists.txt
    │   ├── include/
    │   └── src/
    ├── metadata.json                    # Informações do modelo
    ├── parameters/
    │   └── default.params               # Pesos do modelo
    └── src/
        └── default.relay                # Representação Relay (IR)
```

---

## Adaptação para Outros Modelos

### Para aplicar este processo a outro modelo:

1. **Alterar o Script de Treino (Fase 1)**
   - Ajuste a arquitetura em `create_model()` conforme seu domínio
   - Modifique inputs/outputs conforme necessário
   - Treine com seus dados

2. **Ajustar Input Shapes no Export ONNX (Fase 2)**
   - Em `export_to_onnx.py`, modifique:
   ```python
   input_signature = [
       tf.TensorSpec([1, SEU_TAMANHO_1], tf.float32, name='input1'),
       tf.TensorSpec([1, H, W, C], tf.float32, name='input2')
   ]
   ```

3. **Ajustar Input Shapes no TVMC Compile (Fase 4)**
   - No comando `tvmc compile`, modifique:
   ```bash
   --input-shapes "input1:[1,TAMANHO],input2:[1,H,W,C]"
   ```

4. **TVM permanece o mesmo** (Fase 3)
   - Uma vez instalado, pode reutilizar para qualquer modelo

### Exemplo: Modelo de Classificação de Imagens

```python
# Em export_to_onnx.py
input_signature = [
    tf.TensorSpec([1, 224, 224, 3], tf.float32, name='image')
]

# No tvmc compile
python -m tvm.driver.tvmc compile "$MODEL" \
    --input-shapes "image:[1,224,224,3]" \
    --target "c -keys=cpu" \
    --runtime crt \
    --executor aot \
    --output-format mlf \
    --output "image_classifier_mlf.tar"
```

---

## Troubleshooting

### Problema: LLVM enabled: 0

**Causa:** TVM compilado sem suporte LLVM.

**Solução:**
```bash
cd ~/tvm_aot/build
rm -rf *
cd ..
# Verificar config.cmake tem USE_LLVM="llvm-config"
grep USE_LLVM config.cmake
# Recompilar
cd build && cmake .. && make -j$(nproc)
```

### Problema: ImportError: No module named 'tvm'

**Causa:** Variáveis de ambiente não carregadas.

**Solução:**
```bash
source ~/tvm_aot/.venv/bin/activate
source ~/tvm_aot/enable_tvm.sh
```

### Problema: ONNX model validation error

**Causa:** Shapes incompatíveis ou modelo corrompido.

**Solução:**
```python
import onnx
model = onnx.load('seu_modelo.onnx')
onnx.checker.check_model(model)
# Verificar mensagem de erro específica
```

### Problema: tvmc compile falha com erro de target

**Causa:** Flags incorretas ou incompatíveis.

**Solução:**
- Verificar se `--input-shapes` corresponde ao modelo
- Remover flags não reconhecidas pelo TVM 0.15.0
- Usar exatamente: `--target "c -keys=cpu"` (não `--target-host`)

---

## Referências

- **TVM Documentação:** https://tvm.apache.org/docs/
- **ONNX:** https://onnx.ai/
- **Model Library Format:** https://tvm.apache.org/docs/arch/model_library_format.html
- **microTVM:** https://tvm.apache.org/docs/topic/microtvm/index.html

---

## Checklist Final

Antes de considerar o processo completo, verifique:

- [ ] Modelo Keras treinado e salvo
- [ ] ONNX exportado e validado
- [ ] TVM compilado com LLVM habilitado (`enabled("llvm")` retorna `1`)
- [ ] MLF gerado com sucesso (arquivo .tar existe)
- [ ] Arquivos C extraídos (default_lib0.c, tvmgen_default.h presentes)
- [ ] metadata.json contém informações corretas de inputs/outputs
- [ ] Arquivos copiados para projeto final
- [ ] Commit no Git realizado

**Status:** ✅ Processo completo documentado

---

*Guia criado em: 09/01/2026*  
*Projeto: DigitalTwin - Predictive Maintenance*  
*Modelo: pump_predictive (Industrial Pump Monitoring)*
