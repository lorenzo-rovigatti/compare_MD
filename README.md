# compare_MD

This very simple tutorial code is a barebone Molecular Dynamics package to simulate a Lennard-Jones system on CPU and GPU. It has *no* optimisations and most of the options are set with macros in the `defs.h` file, something that should be *never* ever done in a real code.

The default parameters are:

* Temperature = 1.5
* Integration time step = 0.001
* Lennard-Jones cut-off radius = 2.5
* Probability of extracting a particle velocity = 0.1
* Try to regenerate the particle velocities every 537 steps

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

## Usage

Both executables take two mandatory arguments (number of particles and number density) and an optional one (length of the simulation in MD steps, defaults to 100000). A few examples:

```bash
$ ./compare_MD 100 0.1          # run a CPU simulation with 100 particles at density 0.1 for 100000 steps
$ ./compare_MD 666 0.2 10000    # run a CPU simulation with 666 particles at density 0.1 for 10000 steps
$ ./compare_MD_CUDA 10000 0.15  # run a GPU simulation with 10000 particles at density 0.15 for 100000 steps 
```

## Output

The layout of the lines printed to the standard output is `step U K T`, where the last three are the potential, kinetic and total energy per particle, respectively.

The initial and final configurations are printed in a format supported by [cogli1](https://sourceforge.net/projects/cogli1/) in the `initial.mgl` and `last.mgl` files. 

## Notes

* The code precision (float or double) are set by macros in `defs.h`.
* We do not subtract the cut-off energy.
* We simulate at constant temperature by using the Andersen thermostat.
* The PRNG seed is set to a constant so that all results are reproducible.
* There are few to no checks on what the CUDA functions return. 
