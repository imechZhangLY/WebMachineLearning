// only wasm backend is enabled
import ortWasm from "onnxruntime-web/wasm";
// only webgl backend is enabled
import ortWebGL from "onnxruntime-web/webgl";
// both webnn, webgpu and wasm backends are enabled
import ortWebGPU from "onnxruntime-web/webgpu";
// both webnn, wasm, webgl and webgpu backends are enabled
import ortAll from "onnxruntime-web/all";
// both webgl and wasm backends are enabled
import ortDefault from "onnxruntime-web";

const ways = {
    wasm: ortWasm,
    webgl: ortWebGL,
    webgpu: ortWebGPU,
    all: ortAll,
    default: ortDefault,
}

function log(message) {
    const logElement = document.querySelector("#output");
    logElement.textContent = message;
}

// use an async context to call onnxruntime functions.
async function run(ort, options) {
    try {
        ort.env.wasm.wasmPaths = 'dist/';
        ort.env.wasm.numThreads = 1;
        // input: a 3x4 matrix, b 4x3 matrix
        // output: c 3x3 matrix = a * b
        const session = await ort.InferenceSession.create('./model.onnx', options);

        const dataA = Float32Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        const dataB = Float32Array.from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]);
        const tensorA = new ort.Tensor('float32', dataA, [3, 4]);
        const tensorB = new ort.Tensor('float32', dataB, [4, 3]);

        // prepare feeds. use model input names as keys.
        const feeds = { a: tensorA, b: tensorB };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const dataC = results.c.data;
        log(`data of result tensor 'c': ${dataC}`);

    } catch (e) {
        log(`failed to inference ONNX model: ${e}.`);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.querySelector("#run").addEventListener("click", () => {
        log("running...");
        const way = document.querySelector("#import-way").value;
        const ort = ways[way];
        const backend = document.querySelector("#backend").value;
        run(ort, { executionProviders: [backend] });
    });
});
