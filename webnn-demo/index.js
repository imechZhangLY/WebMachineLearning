const M = 1000;
const N = 1000;

function generateRandomData(size) {
  const data = new Float32Array(size);

  for (let i = 0; i < size; i++) {
    data[i] = Math.random();
  }

  return data;
}

const A = generateRandomData(M * N);
const B = generateRandomData(N * M);

async function matMulWebNN(A, B) {
    const context = await navigator.ml.createContext({deviceType: 'gpu'});
    const builder = new MLGraphBuilder(context);
    const descA = {dataType: 'float32', dimensions: [M, N], shape: [M, N]};
    const descB = {dataType: 'float32', dimensions: [N, M], shape: [N, M]};

    // Step 1: Create a computational graph calculating `c = a * b`.
    const a = builder.input('a', descA);
    const b = builder.input('b', descB);
    const c = builder.matmul(a, b);

    // Step 2: Compile it into an executable graph.
    const graph = await builder.build({c});

    // Step 3: Bind input and output buffers to the graph and execute.
    descA.usage = MLTensorUsage.WRITE;
    descB.usage = MLTensorUsage.WRITE;
    const tensorA = await context.createTensor(descA);
    const tensorB = await context.createTensor(descB);
    context.writeTensor(tensorA, A);
    context.writeTensor(tensorB, B);
    const tensorC = await context.createTensor({
        dataType: 'float32',
        dimensions: [M, M],
        shape: [M, M],
        usage: MLTensorUsage.READ,
    });
    context.dispatch(graph, {a: tensorA, b: tensorB}, {c: tensorC});
    const results = await context.readTensor(tensorC);

    return new Float32Array(results);
}

function matMulJs(A, B) {
    const C = new Float32Array(M * M);

    for (let i = 0; i < M; i++) {
        for (let j = 0; j < M; j++) {
            let sum = 0;

            for (let k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * M + j];
            }

            C[i * M + j] = sum;
        }
    }

    return C;
}

function log(message) {
    const logElement = document.querySelector("#output");
    logElement.textContent += `\n${message}`;
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelector("#run-webnn").addEventListener("click", async () => {
        log("running webNN...");
        const start = performance.now();
        await matMulWebNN(A, B);
        const end = performance.now();
        log(`Finished in ${end - start} ms`);
    });
    
    document.querySelector("#run-js").addEventListener("click", () => {
        log("running JS...");
        const start = performance.now();
        matMulJs(A, B);
        const end = performance.now();
        log(`Finished in ${end - start} ms`);
    });
});