# AIter Project Notes

## Workflow

1. **Before starting**:
   - Read [session.md](.claude/session.md) to understand current progress and next steps
   - Read [issues.md](.claude/issues.md) for issue status and reproduction steps

2. **During work**:
   - Update `session.md` with progress, hypothesis, and TODO changes
   - Update `issues.md` when issue status or test results change
   - Update `knowledge.md` when discovering new technical insights (architecture patterns, debugging techniques, non-obvious behaviors)

3. **Before ending** (before compacting is triggered OR user exits conversation):
   - Update `session.md` with latest progress and next steps
   - Commit `.claude/*.md` files with a summary of progress

## Build and Run Environment

User-specific settings are in `.claude/user.md` (see `.claude/user.md.example` for template).

### Docker Container
```bash
# Use container name from .claude/user.md or user.md.example
docker exec -it <CONTAINER> bash
```

### Build & Run
Run any Python script that invokes a `@compile_ops` annotated function.
This triggers JIT HIP kernel compilation and loads the compiled `.so` under `aiter/jit/`.

### Clean Rebuild
If kernel code changed, remove related `.so` before running:
```bash
rm aiter/jit/*.so
```

### Example
```bash
# Inside container
rm -f aiter/jit/*.so && python -m pytest op_tests/test_mha_fp8.py -v
```

## Technical Documentation

- [Session Notes](.claude/session.md) - Current focus, next steps, and TODO
- [Issue Tracker](.claude/issues.md) - Issue status, reproduction steps, and test results
- [Technical Knowledge](.claude/knowledge.md) - Pipeline design, lane mapping, and debugging insights

## Project Structure

- `aiter/` - Python library source
- `csrc/` - C++/CUDA kernel implementations
- `op_tests/` - Operator tests
- `3rdparty/composable_kernel/` - AMD Composable Kernel library (submodule)

## Best Practices

- **Print all values:** When comparing kernel registers with reference, print all thread_buffer values, not just the first few. Matching on a few elements doesn't mean all match. Do this by hardcoding the lane ID and repeating the cycle (update lane ID → run test → collect output) until you have enough data to compare.
- **Verify and confirm before moving on:** When debugging intermediate values (sp_compute, m, l, P, V, o_acc), verify ALL rows/cols before concluding correctness. Before moving to verify the next stage's intermediate values, always ask the user if the current conclusion is correct. Do not proceed without explicit user confirmation.
- **Ask user for direction:** Before moving to the next debugging step, present options to the user and let them decide the direction. Do not assume what to check next.
- **Batch kernel changes:** Debug prints require expensive recompilation. Plan all changes at once rather than incrementally.
- **Only record verified facts:** Do not write conclusions or hypotheses in this document until they are verified by experiments. Record experiment results and observations, not speculation.

## JIT Compilation Mechanism

The aiter project uses a decorator-based JIT compilation system to build HIP kernels on-demand.

### Flow Overview (using `mha_fwd()` as example)

```
Python call: mha_fwd(q, k, v, ...)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ @compile_ops decorator (aiter/jit/core.py:795)                  │
│  1. Call gen_func() to get md_name and blob_gen_cmd             │
│  2. Try get_module(md_name) to load existing .so                │
│  3. If ModuleNotFoundError → trigger build_module()             │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ build_module() (aiter/jit/core.py:523)                          │
│  1. Read build config from optCompilerConfig.json               │
│  2. Execute blob_gen_cmd (e.g., CK's generate.py) to create     │
│     kernel instantiation source files                           │
│  3. Copy/rename .cpp → .cu in build directory                   │
│  4. Call _jit_compile()                                         │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ _jit_compile() (aiter/jit/utils/cpp_extension.py:1133)          │
│  1. Version check (skip rebuild if unchanged)                   │
│  2. Hipify sources (CUDA → HIP translation)                     │
│  3. _write_ninja_file_and_build_library() → invoke hipcc        │
│  4. Output: {md_name}.so in aiter/jit/                          │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ get_module(md_name) (aiter/jit/core.py:506)                     │
│  1. importlib.import_module() loads the compiled .so            │
│  2. getattr(module, fc_name) retrieves the kernel function      │
│  3. Return kernel op for execution                              │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
  Kernel launch: op(*args, **kwargs)
```

### Key Components

| File | Purpose |
|------|---------|
| `aiter/jit/core.py` | `@compile_ops` decorator, `build_module()`, `get_module()` |
| `aiter/jit/optCompilerConfig.json` | Build config per module (sources, flags, includes) |
| `aiter/jit/utils/cpp_extension.py` | `_jit_compile()`, ninja build, hipify integration |
| `aiter/ops/mha.py` | Example: `mha_fwd()` with `@compile_ops` and `cmdGenFunc_mha_fwd()` |

### Example: mha_fwd()

```python
# aiter/ops/mha.py:192-198
@compile_ops(
    "module_mha_fwd",           # base module name
    fc_name="mha_fwd",          # function to load from .so
    gen_func=cmdGenFunc_mha_fwd, # generates md_name and blob_gen_cmd
    gen_fake=gen_mha_fwd_fake_tensors,
)
def mha_fwd(q, k, v, ...): ...
```

The `cmdGenFunc_mha_fwd()` inspects input dtypes and options to construct:
- `md_name`: e.g., `"mha_fwd_fp8bf16_nbias_nmask_lse_ndropout_nqscale"`
- `blob_gen_cmd`: e.g., `"3rdparty/composable_kernel/.../generate.py -d fwd --receipt 100 --filter ..."`

### JIT Output and Intermediate Files

#### Directory Structure
```
aiter/jit/
├── {md_name}.so                    ← FINAL OUTPUT (Python-loadable shared library)
└── build/
    └── {md_name}/
        ├── blob/                   ← Generated kernel sources
        └── build/                  ← Compilation intermediates
            ├── build.ninja         ← Ninja build file
            ├── *.cuda.o            ← Compiled object files
            ├── *.s                 ← Assembly (with --save-temps)
            ├── *.bc                ← LLVM bitcode
            ├── *.hipfb             ← HIP fatbinary (device code)
            └── {md_name}.so        ← Linked output (copied to aiter/jit/)
```

#### Stage 1: Blob Generation (`blob/`)
Generated by CK's `generate.py` based on input dtype/options:

| File | Purpose |
|------|---------|
| `fmha_fwd_api.cpp` | API dispatcher - selects kernel variant at runtime |
| `fmha_fwd_d128_fp8bf16_batch_*.cpp` | Kernel instantiation for specific tile shape/arch |

Each kernel file contains template instantiation with specific parameters:
- Tile shape (e.g., `256x64x128x128x64x128`)
- Pipeline variant (e.g., `QRKSVS_ASYNC_TRLOAD_V3`)
- Target arch (e.g., `gfx950`, `gfx9`)

#### Stage 2: Source Files Compiled
```
# From blob/ (generated per-variant)
fmha_fwd_api.cpp
fmha_fwd_d128_fp8bf16_batch_*.cpp     (~20 kernel variants)

# From csrc/ (fixed sources)
csrc/cpp_itfs/mha_fwd.cu              # C++ interface
csrc/kernels/mha_common.cu            # Common utilities
csrc/py_itfs_ck/mha_fwd_kernels.cu    # Kernel dispatch
csrc/pybind/mha_fwd_pybind.cu         # Python bindings
```

#### Module Name Derivation
The module name encodes input configuration:
```
mha_fwd_fp8bf16_nbias_nmask_nlse_ndropout_pertensor
        │       │     │     │    │        └── per-tensor quantization
        │       │     │     │    └── no dropout
        │       │     │     └── no LSE output
        │       │     └── no mask (not causal)
        │       └── no bias
        └── FP8 input, BF16 output
```

### Finding Compilation Options

#### Quick Reference

| What | Where | Command |
|------|-------|---------|
| Base config | `optCompilerConfig.json` | `cat aiter/jit/optCompilerConfig.json \| python3 -m json.tool \| grep -A 25 '"module_name"'` |
| Dynamic config | `gen_func` in Python | Look at `cmdGenFunc_*` in `aiter/ops/*.py` |
| Final merged flags | `build.ninja` | `cat aiter/jit/build/{md_name}/build/build.ninja` |
| Default HIP flags | `aiter/jit/core.py:559-610` | Hardcoded in `build_module()` |

#### Flag Sources (merged in order)

1. **Hardcoded defaults** (`core.py:559-610`):
   ```
   -O3, -std=c++20
   -D__HIP_PLATFORM_AMD__=1, -DUSE_PROF_API=1
   -mllvm --amdgpu-kernarg-preload-count=16
   ROCm version-specific flags (checked dynamically)
   ```

2. **From `optCompilerConfig.json`**:
   ```json
   {
     "module_mha_fwd": {
       "srcs": ["..."],
       "flags_extra_cc": ["-DFAV2_ON=1"],
       "flags_extra_hip": ["-DCK_TILE_FMHA_FWD_FAST_EXP2=1", "--save-temps"],
       "extra_include": ["{CK_DIR}/example/ck_tile/01_fmha"],
       "extra_ldflags": "None",
       "blob_gen_cmd": ["{CK_DIR}/.../generate.py -d fwd --receipt 600 --output_dir {}"]
     }
   }
   ```

3. **From `gen_func` return dict** (runtime):
   - Can override `md_name`, `blob_gen_cmd`
   - Merged via `d_args.update(custom_build_args)`

#### Example: View mha_fwd Config
```bash
# Static config
cat aiter/jit/optCompilerConfig.json | python3 -m json.tool | grep -A 25 '"module_mha_fwd"'

# Final compiled flags (after build)
head -12 aiter/jit/build/mha_fwd_fp8bf16_nbias_nmask_nlse_ndropout_pertensor/build/build.ninja
```

## CK Codegen System

CK kernels are heavily templated with dozens of parameters. Codegen enables parallel compilation by generating one .cpp per kernel variant.

### Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CK SIDE                                         │
│  3rdparty/composable_kernel/example/ck_tile/01_fmha/                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  generate.py                    Entry point                                  │
│  codegen/ops/fmha_fwd.py       Kernel enumeration + C++ templates           │
│  fmha_fwd.hpp                  PUBLIC API: fmha_fwd_traits, fmha_fwd_args   │
│                                                                              │
│  OUTPUT: blob/*.cpp            Generated kernel instantiations               │
│          fmha_fwd_api.cpp      Runtime dispatch logic                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             AITER SIDE                                       │
│  aiter/jit/                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  optCompilerConfig.json        Specifies blob_gen_cmd to invoke CK's        │
│                                generate.py with receipt filters              │
│  core.py                       @compile_ops decorator, build_module()        │
│                                                                              │
│  csrc/cpp_itfs/mha_fwd.cu      Aiter wrapper calling CK's fmha_fwd()        │
│  csrc/py_itfs_ck/mha_fwd_kernels.cu  PyTorch tensor → fmha_fwd_args         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### CK Codegen Classes (`codegen/ops/fmha_fwd.py`)

| Class | Purpose |
|-------|---------|
| `FmhaFwdTileSize` | Block/warp tile dimensions (bm0, bn0, bk0, warp counts) |
| `FmhaFwdPipeline` | Feature flags (vlayout, padding, mask, bias, dropout, qscale) |
| `FmhaFwdKernel` | Combines arch + dtype + tile + pipeline → generates .cpp |
| `FmhaFwdApiPool` | Collects all traits for API dispatcher generation |

### CK Public API (`fmha_fwd.hpp`)

```cpp
// Runtime arguments
struct fmha_fwd_args {
    const void* q_ptr, *k_ptr, *v_ptr;
    const void* q_descale_ptr, *k_descale_ptr, *v_descale_ptr;  // FP8 scales
    void* lse_ptr, *o_ptr;
    ck_tile::index_t seqlen_q, seqlen_k, batch, hdim_q, hdim_v, nhead_q, nhead_k;
    float scale_s, logits_soft_cap;
    // strides, mask info, dropout...
};

// Dispatch traits
struct fmha_fwd_traits {
    int hdim_q, hdim_v;
    std::string data_type;       // "fp16", "bf16", "fp8bf16"
    bool is_group_mode;
    mask_enum mask_type;
    bias_enum bias_type;
    quant_scale_enum qscale_type;  // no_scale, pertensor
    // ...
};

// Entry point
float fmha_fwd(fmha_fwd_traits, fmha_fwd_args, const ck_tile::stream_config&);
```

### Generated Kernel Structure

Each `blob/*.cpp` contains:
```cpp
#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__gfx950__))

using fmha_dtype = FmhaFwdFp8Bf16;
using fmha_block_tile = ck_tile::sequence<256, 64, 128, 128, 64, 128>;
using fmha_pipeline = ck_tile::BlockFmhaFwdV3Pipeline<...>;
using fmha_kernel = ck_tile::FmhaFwdV3Kernel<fmha_pipeline, fmha_epilogue>;

// Explicit template specialization
using trait = fmha_fwd_traits_<128, FmhaFwdFp8Bf16, false, 256, 64, ...>;

template<>
float fmha_fwd_<trait, ck_tile::gfx950_t>(const ck_tile::stream_config& s, fmha_fwd_args a) {
    auto [kargs, grids] = fmha_fwd_v3_create_kargs_and_grids<k_>(a);
    return ck_tile::launch_kernel(...);
}
#endif
```

### Generated API Dispatcher (`fmha_fwd_api.cpp`)

```cpp
float fmha_fwd(fmha_fwd_traits traits, fmha_fwd_args args, ...) {
    // V3 pipeline check (gfx950 + fp8bf16 + pertensor + constraints)
    if (can_dispatch_v3) return fmha_fwd_v3(traits, args, config);
    else return fmha_fwd_v2(traits, args, config);
}

// Nested dispatch: arch → dtype → hdim → (mode, vlayout, mask, bias, ...)
float fmha_fwd_v2(...) {
    if(device_name == "gfx950") {
        if(t.data_type == "fp8bf16") {
            if(t.hdim_q <= 128) {
                using trait_ = fmha_fwd_traits_<128, FmhaFwdFp8Bf16, ...>;
                return fmha_fwd_<trait_, ck_tile::gfx950_t>(s, a);
            }
        }
    }
}
```

### Receipt System (Product Filters)

| Receipt | Product | Filter Criteria |
|---------|---------|-----------------|
| 100 | Aiter mha_fwd | fp16/bf16/fp8bf16, batch mode, row-major V |
| 200 | Aiter mha_varlen_fwd | Group mode variant |
| 600 | Aiter C++ API | Both batch and group modes |
| 2-3 | Flash Attention | fp16/bf16, no skip |
| 800 | fp32 only | All variations |

### Aiter Integration Points

#### 1. Tensor → CK Args (`csrc/py_itfs_ck/mha_fwd_kernels.cu`)
```cpp
mha_fwd_args get_ck_fmha_fwd_args(..., const at::Tensor q, ...) {
    // Extract strides from PyTorch tensors
    ck_tile::index_t stride_q = q.stride(1);
    return mha_fwd_args{q.data_ptr(), k.data_ptr(), ...};
}
```

#### 2. Dispatch to CK (`csrc/cpp_itfs/mha_fwd.cu`)
```cpp
float mha_fwd(mha_fwd_args args, const ck_tile::stream_config& s) {
#if FAV3_ON
    ret = fmha_fwd_v3(args, s);   // Try V3 pipeline first
#endif
#if FAV2_ON
    if(ret == -1) ret = fmha_fwd_ck(args, s);  // Fall back to V2
#endif
    return ret;
}
```

### Complete Flow Example

```
Python: mha_fwd(q, k, v, ...)  # q is fp8, out is bf16
       │
       ▼
@compile_ops → cmdGenFunc_mha_fwd()
       │ Returns: md_name = "mha_fwd_fp8bf16_nbias_nmask_nlse_ndropout_pertensor"
       │          blob_gen_cmd = "generate.py -d fwd --receipt 100 ..."
       ▼
build_module() → executes blob_gen_cmd
       │
       ▼
CK generate.py → writes to blob/
       │ ├── fmha_fwd_api.cpp
       │ └── fmha_fwd_d128_fp8bf16_batch_b256x64x..._gfx950.cpp (×N variants)
       ▼
hipcc compiles all → mha_fwd_fp8bf16_....so
       │
       ▼
Python imports .so, calls mha_fwd()
       │
       ▼
C++: get_ck_fmha_fwd_args() → fmha_fwd_args
       │
       ▼
CK: fmha_fwd(traits, args, stream)
       │ Dispatches based on traits to correct kernel variant
       ▼
GPU: FmhaFwdV3Kernel launches with optimal tile/pipeline
```

## Key Files

### FP8 Flash Attention
- `csrc/ck_fmha_fwd/src/fmha_fwd_kernel.hpp` - Kernel entry point, scale definitions
- `3rdparty/composable_kernel/.../block_fmha_pipeline_qr_ks_vs_async.hpp` - Online softmax implementation
- `op_tests/test_mha_fp8.py` - FP8 attention tests and reference implementation
