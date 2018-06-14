# compare_MD

## Requirements

* A C compiler (preferable gcc)
* cmake
* make
* Optional: the CUDA libs to enable CUDA support

## Compilation

```
mkdir build
cd build
cmake ..
make -j4
```

The compare_MD executable will be placed in the `build/bin` folder.

## Notes

* We do not subtract the cut-off energy
* The RNG seed is set to a constant so that all results are reproducible.
