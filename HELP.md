# TornadoVM

## Installation (Windows, OpenCL)

Download the release:

```cmd
curl -L -o tornadovm-7.3.1-jdk22plus-opencl-windows-amd64.zip https://github.com/beehive-lab/TornadoVM/releases/download/v7.3.1/tornadovm-7.3.1-jdk22plus-opencl-windows-amd64.zip
```

Extract it:

```cmd
tar -xf tornadovm-7.3.1-jdk22plus-opencl-windows-amd64.zip
```

Set the environment variables:

```cmd
set TORNADOVM_HOME=%cd%\tornadovm-7.3.1-jdk22plus-opencl
set PATH=%TORNADOVM_HOME%\bin;%PATH%
```

## Running tests

Use your own model path via `-Dmodel=<path>`, e.g. `-Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf`.

### Vector multiplication CPU

```bash
mvn clean test -Dtest=TornadoVecMulTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dchunk=16384 -Dextra.jvm.flags="-XX:-UseSuperWord" -Dtornado.device.memory=8GB -Duse.tornadovm=true
```
### Vector multiplication GPU

```bash
mvn clean test -Dtest=TornadoVecMulTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dtornado.device.memory=8GB -Duse.tornadovm=true
```

### JIT LLM question and answer

```bash
mvn clean test -Dtest=TornadoJitLLMTest#questionAndAnswerTest -Dstyle.color=always -Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=8GB -Duse.tornadovm=true
```

### Vector multiplication benchmark CPU

```bash
mvn clean test -Dtest=TornadoVecMulBenchmarkTest#vecMulBenchmarkTest -Dstyle.color=always -Dsize=10000000000 -Dchunk=16384 -Dextra.jvm.flags="-XX:-UseSuperWord" -Dtornado.device.memory=8GB -Duse.tornadovm=true
```

### Vector multiplication benchmark GPU

```bash
mvn clean test -Dtest=TornadoVecMulBenchmarkTest#vecMulBenchmarkTest -Dstyle.color=always -Dsize=10000000000 -Dtornado.device.memory=8GB -Duse.tornadovm=true
```

### Fraud scoring

```bash
mvn clean test -Dtest=TornadoFraudScoringTest#fraudScoringPositiveTest -Dstyle.color=always -Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=8GB -Dfraud=true -Duse.tornadovm=true
```

### Alert explainer

```bash
mvn clean test -Dtest=TornadoFraudAlertExplainerTest#explainsAlertTest -Dstyle.color=always -Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=8GB -Dfraud=true -Duse.tornadovm=true
```