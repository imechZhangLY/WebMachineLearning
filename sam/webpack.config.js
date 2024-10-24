import CopyWebpackPlugin from 'copy-webpack-plugin';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * @type {import('webpack').Configuration}
 */
export default {
    mode: 'development',
    devtool: 'source-map',
    entry: {
        'dist/index': './index.js',
    },
    output: {
        filename: '[name].js',
        path: __dirname,
        library: {
            type: 'module',
        },
    },
    plugins: [
        // Copy .wasm files to dist folder
        new CopyWebpackPlugin({
            patterns: [
                {
                    from: 'node_modules/onnxruntime-web/dist/*.wasm',
                    to: 'dist/[name][ext]'
                },
            ],
        }),
    ],
    devServer: {
        static: {
            directory: __dirname
        },
        headers: {
            "Cross-Origin-Opener-Policy": "same-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
        },
        port: 8080
    },
    experiments: {
        outputModule: true,
    },
};
