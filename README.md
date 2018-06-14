# compare_MD

This very simple tutorial code is a barebone Molecular Dynamics package to simulate a Lennard-Jones system on CPU and GPU. It has *no* optimisations and most of the options are set with macros in the `defs.h` file, something that should be *never* ever done in a real code.

## Requirements

* A c++ compiler (g++ recommended)
* cmake
* make
* Optional: CUDA libs to enable CUDA support

## Compilation

```
mkdir build
cd build
cmake ..
make -j4
```

At the end of the compilation stage the `compare_MD` executable can be found in the `build/bin` folder. If CUDA libraries are found by `cmake`, then the `compare_MD_CUDA` will be compiled as well.

## Notes

* We do not subtract the cut-off energy
* The RNG seed is set to a constant so that all results are reproducible.
