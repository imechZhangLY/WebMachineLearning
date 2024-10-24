import ort from "onnxruntime-web/all";

ort.env.debug = true;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "dist/";

const MODELS = {
    sam_b: {
        encoder: {
            name: "sam-b-encoder",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.encoder-fp16.onnx",
            size: 180,
        },
        decoder: {
            name: "sam-b-decoder",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b_01ec64.decoder.onnx",
            size: 17,
        },
    },
    sam_b_int8: {
        encoder: {
            name: "sam-b-encoder-int8",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b-encoder-int8.onnx",
            size: 108,
        },
        decoder: {
            name: "sam-b-decoder-int8",
            url: "https://huggingface.co/schmuell/sam-b-fp16/resolve/main/sam_vit_b-decoder-int8.onnx",
            size: 5,
        },
    },
};

const MODEL_WIDTH = 1024;
const MODEL_HEIGHT = 1024;

function log(i) {
    document.getElementById('status').innerText += `\n${i}`;
}

async function hasFp16() {
    try {
        const adapter = await navigator.gpu.requestAdapter()
        const features = adapter.features.keys()
        for (const feature of features) {
            console.log(feature, adapter.features.has(feature))
        }
        return adapter.features.has('shader-f16')
    } catch (e) {
        return false
    }
}

async function fetchAndCache(url) {
    try {
        const cache = await caches.open("onnx");
        let cachedResponse = await cache.match(url);
        if (cachedResponse == undefined) {
            await cache.add(url);
            cachedResponse = await cache.match(url);
        }
        const data = await cachedResponse.arrayBuffer();
        return data;
    } catch (error) {
        return await fetch(url).then(response => response.arrayBuffer());
    }
}

async function prepareInputFeed(img) {
    const canvas = new OffscreenCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, img.width, img.height);

    const imageData = ctx.getImageData(0, 0, img.width, img.height);
    const imageTensor = await ort.Tensor.fromImage(imageData, { resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT });
    // const imageTensor = await ort.Tensor.fromImage(img, { resizedWidth: MODEL_WIDTH, resizedHeight: MODEL_HEIGHT });

    return { "input_image": imageTensor };
}

function cloneTensor(t) {
    return new ort.Tensor(t.type, Float32Array.from(t.data), t.dims);
}

function prepareInputFeedForDecoder(encoderResults, point) {
    const maskInput = new ort.Tensor(new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMask = new ort.Tensor(new Float32Array([0]), [1,]);
    const originalImageSize = new ort.Tensor(new Float32Array([MODEL_HEIGHT, MODEL_WIDTH]), [2,]);
    const pointCoords = new ort.Tensor(new Float32Array(point), [1, 1, 2]);
    const pointLabels = new ort.Tensor(new Float32Array([1]), [1, 1]);

    const result = {
        "image_embeddings": cloneTensor(encoderResults.image_embeddings),
        "point_coords": pointCoords,
        "point_labels": pointLabels,
        "mask_input": maskInput,
        "has_mask_input": hasMask,
        "orig_im_size": originalImageSize
    }

    return result;
}


async function loadModels() {
    log("Loading models");
    // 1. Check users' browser environment and choose a suitable model
    const modelName = await hasFp16() ? "sam_b" : "sam_b_int8";
    const backends = ["webgpu", "webgl", "wasm"];

    // 2. Download and fetch the model
    const model = MODELS[modelName];
    const encoderModelBuffer = await fetchAndCache(model.encoder.url);
    const decoderModelBuffer = await fetchAndCache(model.decoder.url);

    // 3. Create an InferenceSession for the model
    const encoderSession = await ort.InferenceSession.create(encoderModelBuffer, { executionProviders: backends });
    const decoderSession = await ort.InferenceSession.create(decoderModelBuffer, { executionProviders: backends });

    log("Models loaded");
    return { encoderSession, decoderSession };
}

function getPoint(event) {
    let x = event.clientX;
    let y = event.clientY;
    const img = document.querySelector("#input_img");
    const rect = img.getBoundingClientRect();

    x = (x - rect.left);
    y = (y - rect.top);
    return [x, y];
}

async function processOutput(mask, img) {
    const [w, h] = [img.width, img.height];
    const maskImageData = mask.toImageData();
    console.log(mask, maskImageData);

    const imageCanvas = new OffscreenCanvas(w, h);
    const imageContext = imageCanvas.getContext('2d');
    imageContext.drawImage(img, 0, 0, w, h);
    const imagePixelData = imageContext.getImageData(0, 0, w, h);

    const cutCanvas = new OffscreenCanvas(w, h);
    const cutContext = cutCanvas.getContext('2d');
    const cutPixelData = cutContext.getImageData(0, 0, w, h);

    const maskCanvas = new OffscreenCanvas(w, h);
    const maskContext = maskCanvas.getContext('2d');
    maskContext.drawImage(await createImageBitmap(maskImageData), 0, 0);
    const maskPixelData = maskContext.getImageData(0, 0, w, h);

    for (let i = 0; i < maskPixelData.data.length; i += 4) {
        if (maskPixelData.data[i] > 0) {
            for (let j = 0; j < 4; ++j) {
                const offset = i + j;
                cutPixelData.data[offset] = imagePixelData.data[offset];
            }
        }
    }
    cutContext.putImageData(cutPixelData, 0, 0);

    document.querySelector("#output_img").src = URL.createObjectURL(await cutCanvas.convertToBlob());
}

async function main() {
    const { encoderSession, decoderSession } = await loadModels();

    // 4. Prepare input feeds
    let feeds;
    let encoderResults;
    const img = document.querySelector("#input_img");
    if (img.complete) {
        log("Input image loaded");
        feeds = await prepareInputFeed(img);
    } else {
        img.addEventListener("load", async () => {
            log("Input image loaded");
            feeds = await prepareInputFeed(img);
        });
    }

    img.addEventListener("click", async (e) => {
        log("Image Clicked");
        if (encoderResults == undefined) {
            console.error("Please run the encoder first");
            return;
        }
        // 4. Prepare input feeds for the decoder
        const point = getPoint(e);
        log(`Point: ${point}`);
        const decoderFeed = prepareInputFeedForDecoder(encoderResults, point);

        // 5. Run the decoder model
        log("Running the decoder model");
        const start = performance.now();
        const decoderResults = await decoderSession.run(decoderFeed);
        const end = performance.now();
        log(`Decoder finished in ${end - start} ms`);

        // 6. Process the decoder results
        const mask = decoderResults.masks;
        await processOutput(mask, img);
    });

    document.querySelector("#run").addEventListener("click", async () => {
        if (feeds == undefined) {
            console.error("Please load an image first");
            return;
        }

        log("Running the model");
        // 5. Run the model
        const start = performance.now();
        encoderResults = await encoderSession.run(feeds);
        const end = performance.now();
        log(`Encoder finished in ${end - start} ms`);
    });
}

main();
