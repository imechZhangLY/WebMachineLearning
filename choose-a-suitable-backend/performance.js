// both webnn, wasm, webgl and webgpu backends are enabled
import ort from "onnxruntime-web/all";

function log(message) {
    const logElement = document.querySelector("#output");
    logElement.textContent = message;
}

const generateRandomData = (size) => {
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.random();
    }
    return data;
};
const M = 4000;
const N = 4000;
const A = generateRandomData(M * N);
const B = generateRandomData(N * M);
// use an async context to call onnxruntime functions.
async function run(ort, options) {
    try {
        ort.env.wasm.wasmPaths = 'dist/';
        ort.env.wasm.numThreads = 1;
        ort.env.logLevel = 'verbose';
        ort.env.debug = true;
        const session = await ort.InferenceSession.create('./mat_mul.onnx', options);

        const tensorA = new ort.Tensor('float32', A, [M, N]);
        const tensorB = new ort.Tensor('float32', B, [N, M]);

        // prepare feeds. use model input names as keys.
        const feeds = { A: tensorA, B: tensorB };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        // const dataC = results.C.data;
        // console.log(dataC);
        log("successfully run ONNX model");
    } catch (e) {
        log(`failed to inference ONNX model: ${e}.`);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelector("#run").addEventListener("click", async () => {
        log("running...");
        const backend = document.querySelector("#backend").value;
        const start = performance.now();
        await run(ort, { executionProviders: [backend] });
        const end = performance.now();
        // log(`Finished in ${end - start} ms`);
        document.querySelector(`#${backend}`).textContent = `${end - start} ms`;
    });
});