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

### Vector multiplication benchmark

```bash
mvn clean -Dtest=TornadoVecMulTest#vecMulBenchmarkTest -Dstyle.color=always -Dsize=10000000000 -Duse.tornadovm=false test
```

### JIT LLM question and answer

```bash
mvn clean -Dtest=TornadoJitLLMTest#questionAndAnswerTest -Dstyle.color=always -Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf -Duse.tornadovm=false test
```

### Fraud scoring

```bash
mvn clean -Dtest=TornadoFraudScoringTest#fraudScoringPositiveTest -Dstyle.color=always -Dmodel=/Users/zura/Downloads/Qwen3-0.6B-Q8_0.gguf -Dfraud=true -Duse.tornadovm=false test
```

### Fraud alert explainer

```bash
mvn clean -Dtest=TornadoFraudAlertExplainerTest#explai