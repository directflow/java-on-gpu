# TornadoVM

## Installation (Windows, OpenCL)

Download the release:

```cmd
wget https://github.com/beehive-lab/TornadoVM/releases/download/v7.3.1/tornadovm-7.3.1-jdk22plus-cuda-linux-amd64.zip
```

Extract it:

```cmd
unzip tornadovm-7.3.1-jdk22plus-opencl-linux-amd64.zip
```

Set the environment variables:

```cmd
export TORNADOVM_HOME="$(pwd)/tornadovm-7.3.1-jdk22plus-cuda"
export PATH=$TORNADOVM_HOME/bin:$PATH
```

## Running tests

Use your own model path via `-Dmodel=<path>`, e.g. `-Dmodel=$HOME/models/Qwen3-0.6B-Q8_0.gguf`.


### CPU Vector multiplication

```bash
mvn clean test -Dtest=CpuVecMulTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dchunk=16384
```

### VectorAPI Vector multiplication

```bash
mvn clean test -Dtest=VectorApiVecMulTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dchunk=16384 -Dextra.jvm.flags="-XX:-UseSuperWord"
```

### TornadoVM Vector multiplication

```bash
mvn clean test -Dtest=TornadoVecMulTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dtornado.device.memory=16GB -Duse.tornadovm=true
```

### TornadoVM Vector multiplication wrong

```bash
mvn clean test -Dtest=TornadoVecMulWrongTest#vecMulTest -Dstyle.color=always -Dsize=10000000000 -Dtornado.device.memory=16GB -Duse.tornadovm=true
```

### JIT LLM question and answer

```bash
mvn clean test -Dtest=TornadoJitLLMTest#questionAndAnswerTest -Dstyle.color=always -Dmodel=$HOME/models/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=16GB -Duse.tornadovm=true
```

### Vector multiplication benchmark CPU

```bash
mvn clean test -Dtest=TornadoVecMulBenchmarkTest#vecMulBenchmarkTest -Dstyle.color=always -Dsize=10000000000 -Dchunk=16384 -Dextra.jvm.flags="-XX:-UseSuperWord" -Dtornado.device.memory=16GB -Duse.tornadovm=true
```

### Vector multiplication benchmark GPU

```bash
mvn clean test -Dtest=TornadoVecMulBenchmarkTest#vecMulBenchmarkTest -Dstyle.color=always -Dsize=10000000000 -Dtornado.device.memory=16GB -Duse.tornadovm=true
```

### Fraud scoring

```bash
mvn clean test -Dtest=TornadoFraudScoringTest#fraudScoringPositiveTest -Dstyle.color=always -Dmodel=$HOME/models/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=16GB -Dfraud=true -Duse.tornadovm=true
```

### Alert explainer

```bash
mvn clean test -Dtest=TornadoFraudAlertExplainerTest#explainsAlertTest -Dstyle.color=always -Dmodel=$HOME/models/Qwen3-0.6B-Q8_0.gguf -Dtornado.device.memory=16GB -Dfraud=true -Duse.tornadovm=true
```


### Babylon Vector multiplication

```bash
H=~/babylon/hat/build
CP=$(ls $H/hat-*.jar | grep -v example | tr '\n' ':')
M=$(find ~/.m2 \( -name "slf4j-api-2.0.20.jar" -o -name "tornado-api-7.3.1-jdk22plus.jar" -o -name "logback-classic-1.6.5.jar" -o -name "logback-core-1.6.5.jar" \) | tr '\n' ':')
R=$(java -version 2>&1 | head -1 | sed -E 's/[^"]*"([0-9]+).*/\1/')
```

```bash
javac --enable-preview --release $R --add-modules jdk.incubator.code -cp "$CP:$M" \
  -sourcepath src/main/java -d out \
  src/main/java/ge/bog/gpu/babylon/BabylonVecMul.java
```

```bash
HAT=INFO java --enable-preview --add-modules jdk.incubator.code --enable-native-access=ALL-UNNAMED \
  -Djava.library.path=$H -Dsize=1000000 -cp "out:$CP:$M" ge.bog.gpu.babylon.BabylonVecMul
```

### Babylon Vector multiplication oracle

```bash
H=~/babylon/hat/build
CP=$(ls $H/hat-*.jar | grep -v example | tr '\n' ':')
M=$(find ~/.m2 \( -name "slf4j-api-2.0.20.jar" -o -name "tornado-api-7.3.1-jdk22plus.jar" -o -name "logback-classic-1.6.5.jar" -o -name "logback-core-1.6.5.jar" \) | tr '\n' ':')
R=$(java -version 2>&1 | head -1 | sed -E 's/[^"]*"([0-9]+).*/\1/')
```

```bash
rm -rf out
javac --enable-preview --release $R --add-modules jdk.incubator.code,jdk.incubator.vector \
  -cp "$CP:$M" -sourcepath src/main/java -d out \
  src/main/java/ge/bog/gpu/babylon/BabylonVecMul.java \
&& HAT=INFO java --enable-preview --add-modules jdk.incubator.code,jdk.incubator.vector \
  --enable-native-access=ALL-UNNAMED \
  -Djava.library.path=$H -Dsize=10000000000 -cp "out:$CP:$M" ge.bog.gpu.babylon.BabylonVecMul
```
