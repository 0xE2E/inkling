# Building the engine

The engine is the `engine/` tree: Unsloth's llama.cpp build b12588 (which has the `inkling2` architecture and the DSpark
drafter) with the Inkling changes on top. It builds like llama.cpp with the CUDA backend.

## Tested toolchain

- Windows 11, Visual Studio 2022 (MSVC 19.44), CUDA toolkit 13.3, CMake, Ninja
- GPU: RTX 5080 Laptop, 16 GB (compute capability 12.0)

Other compilers, CUDA versions, GPUs and operating systems were not tested.

## Configure and build

From a "x64 Native Tools Command Prompt for VS 2022" in the repository root:

    cmake -S engine -B engine/build -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 ^
      -DGGML_NATIVE=ON ^
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF

    cmake --build engine/build -j 12 --target llama-server llama-bench llama-perplexity

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability (120 for RTX 50-series).

Check that OpenMP was found before measuring anything: the configure log prints `Found OpenMP: TRUE`, and the server's
`system_info` line contains `OPENMP = 1`. A build without OpenMP runs, but the drafter becomes about 20 times slower
(decode fell from ~85 to ~20 tok/s in one such build).

The binaries land in `engine/build/bin`. `start-inkling.ps1` looks there by default; pass `-Bin` or set `INK_BIN` to use
another folder.

## Running

    .\start-inkling.ps1 -Model <Inkling-2-Mini-UD-Q4_K_XL.gguf> -DSpark <dspark-Inkling-2-Mini-Q8_0.gguf>

Both files are in Unsloth's `Inkling-2-Mini-GGUF` repository. Defaults: 128,000-token context, port 8080, host 127.0.0.1.
`-DryRun` prints the command and the switch environment without starting. `-NoDraft` runs without the drafter, `-Stock`
without any engine switch (the measured floor).
