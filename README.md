# [GPU Programming 2022/23] CUDA Path Tracer

![sample render](images/cornell_box_dragon.png "Cornell box with Stanford dragon (~100k triangles) rendered on our path tracer")

This is a path tracer written in CUDA for the _GPU Programming_ course at Politecnico di Torino, developed by Davide Miola and Cristiano Canepari.

## Features

- Full unbiased progressive path tracing of arbitrary triangle meshes
- Scene loading from obj+mtl file combo
- Supports loading camera parameters (like position, orientation and horizontal FoV) from the .obj file (see [Camera parameters](#camera-parameters))
- Perfectly diffuse materials (Lambertian BRDF)
- Real time framebuffer output to screen via CUDA-Vulkan interop
- Encode result to PNG file on exit
- 4x SuperSampling AntiAliasing
- Next-event estimation (a.k.a. explicit light sampling) for faster convergence
- BVH acceleration structure for speeding up ray tracing

The renderer achieves **indistinguishable results compared to production-grade render engines like Blender Cycles** (within the implemented features contraint, i.e. perfectly diffuse materials only), with **comparable performance** (in samples per second).

## Requirements

The program requires an Nvidia GPU compatible with CUDA 10.2 (only if doing Vulkan interop via virtual memory) and Vulkan 1.2 with `VK_KHR_external_memory_win32` (Windows) or `VK_KHR_external_memory_fd` (Linux) for interop via virtual memory, `VK_EXT_external_memory_host` otherwise ("zero copy" memory).

**The Jetson Nano _is_ supported**.

## Compilation

_Note: the Vulkan interface of the app is a Rust library (`cuda-viewer`), which is then linked (dynamically or statically) against the main program by NVCC._

### Native compilation for Windows and Linux hosts

- Compiling the viewer library
    - Install the latest stable Rust toolchain with rustup ([see](https://www.rust-lang.org/learn/get-started))
    - Compile the library:

            cd cuda-viewer
            cargo build --release

        _Note: this will build both a static and a dynamic version of the library._
        
        _Note: **Wayland support is disabled by default**, if you need it, enable the `wayland` cargo feature._

        _Note: if your GPU does not support CUDA 10.2, or is otherwise incompatible with the new virtual memory management API (as is the case with the Jetson Nano), you **must** enable the `zero-copy-mem` cargo feature._

        You can enable cargo features at compile time by appending `--features feature1,feature2,...` to the `cargo build` command.
- Compiling the main app and linking the viewer

    You can choose to link the library statically or dynamically, with the latter method being generally easier, but requiring that you carry around the dynamic library with the main executable at all times.

    - Dynamic linking

        - Windows

                cd app
                nvcc main.cu lodepng.cpp -l"cuda_viewer.dll" -lcuda -I. -L..\cuda-viewer\target\release -use_fast_math -o app.exe

            The required .dll can then be found in `cuda-viewer/target/release/cuda_viewer.dll`
        - Linux
        
                cd app
                nvcc main.cu lodepng.cpp -lcuda_viewer -lcuda -I. -L../cuda-viewer/target/release -use_fast_math -o app

            The required .so can then be found in `cuda-viewer/target/release/libcuda_viewer.so`

    - Static linking

        Static linking is a matter of finding out the static libraries needed by the Rust library and linking them all to the final executable.

        You can find a list of libraries needed on your platform with

            cd cuda-viewer
            cargo rustc --release -- --print native-static-libs

        On _my_ Windows 11 machine this resulted in the following command:
        
            cd app
            nvcc main.cu lodepng.cpp -I. -L..\cuda-viewer\target\release -use_fast_math -lcuda -lcuda_viewer -lkernel32 -luser32 -ladvapi32 -luserenv -lws2_32 -lbcrypt -lmsvcrt -llegacy_stdio_definitions -lOle32 -lWinmm -lGdi32 -luxtheme -limm32 -ldwmapi --linker-options=/NODEFAULTLIB:libcmt.lib -o app.exe

    In any case, if you previously compiled the viewer with the `zero-copy-mem` feature enabled, you **must** define `USE_ZERO_COPY_MEMORY` (with `-DUSE_ZERO_COPY_MEMORY`), and you can remove `-lcuda`.

### Cross compilation for the Jetson Nano

To ease the build process for the Jetson Nano platform, we provide a convenient aarch64 container which can be emulated via QEMU on any modern Linux host.

1. Install Docker
2. Setup Docker to emulate aarch64 containers through QEMU, [see](https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson#enabling-jetson-containers-on-an-x86-workstation-using-qemu)
3. Build the container image:

        docker build -t l4t-builder build-jetson-nano
4. Build the app with static linking:

        docker run --rm -v path/to/project/root:/proj l4t-builder

At this point the compiled program can be found in `path/to/project/root/app/app`, and can be loaded directly on a Jetson Nano.

### Notes on running on the Jetson Nano

Since the Vulkan viewer requires a running X server, you'll need to run your Jetson Nano in standalone mode.

Also, the Nvidia L4T distro running on the Nano has a 5 second limit on every CUDA kernel active by default, which this program is very likely to exceed with any but the simplest scenes. Because of this, you are advised to disable the GPU watchdog with the instructions found [here](https://forums.developer.nvidia.com/t/kernel-launch-timeout/192904/12).

## Compilation switches

Some features can be disabled through defines at compile time:

- `NO_NEXT_EVENT_ESTIMATION`: disables _next event estimation_, a.k.a. _explicit light sampling_, used to explicitly sample lights in the scene, thus greatly speeding up convergence, especially for smaller light sources. Without it, lights can only be reached by randomly bouncing around the scene.
- `NO_BVH`: disables the _Axis Aligned Bounding Box-based Bounding Volume Hierarchy_, which is an acceleration structure used to speed up ray traversal and allow to scale to scenes with millions of triangles.

Other configuration options, like resolution and minimum/maximum number of bounces per light ray, are available in `app/utils.cuh`.

## Camera parameters

The virtual camera that "_takes the picture_" of the scene has a lot of parameters, which can be embedded into the .obj containing the scene's geometry. If no such data is found when loading the model, a default camera is used instead, which is probably not what you want.

The format for specifying these parameters in the following:

```
g cameraParameters
# position
v <x> <y> <z>
# up direction
v <x> <y> <z>
# view direction
v <x> <y> <z>
# horizontal FoV
v <hfov> <doesn't matter> <doesn't matter>
p -4
p -3
p -2
p -1
```

Put the above snippet at the end of your .obj and substitute the placeholders inside angle brackets with the values of your virtual camera (see provided scenes in `app/scenes` for examples).

## Repository structure

| Folder | Contents |
| - | - |
| `app` | Main CUDA path tracer source |
| `app/scenes` | Sample scenes |
| `app/scenes/blend` | Blender projects for some sample scenes. Just open them and press F12 to render the same image you'd get with our path tracer! |
| `build-jetson-nano` | Dockerfile for building for Jetson Nano |
| `cuda-viewer` | Rust crate for Vulkan-based viewer |
| `images` | Images embedded in this README |
