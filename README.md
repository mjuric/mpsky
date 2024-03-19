# Minor Planet Sky (mpsky)

Quickly find the minor planets present in a certain field of view at a certain time.

[![PyPI](https://img.shields.io/pypi/v/mpsky?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/mpsky/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/mjuric/mpsky/smoke-test.yml)](https://github.com/mjuric/mpsky/actions/workflows/smoke-test.yml)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/mjuric/mpsky/asv-main.yml?label=benchmarks)](https://mjuric.github.io/mpsky/)

<!-- [![Codecov](https://codecov.io/gh/mjuric/mpsky/branch/main/graph/badge.svg)](https://codecov.io/gh/mjuric/mpsky)
[![Read The Docs](https://img.shields.io/readthedocs/mpsky)](https://mpsky.readthedocs.io/) -->

## Quick start

It's easiest to install this package from PyPI:
```
pip install mpsky
```
and then use it to query for minor planets present in a field of view using:
```
mpsky query <mjd> <ra_deg> <dec_deg> --radius=<radius_deg>
```
, for example:
```
mpsky query 60853.1 32 11 --radius=1.8
```
. This will query the ephemerides from the default server (currently https://sky.dirac.dev).

## Building your own ephemerides cache and running an API service

This all assumes the code is run on UW's epyc machine (it's TBD to generalize these instructions; in the meantime, we hope you can figure it out yourself).

To build the ephemerides cache, run something like:
```
mpsky build /astro/store/epyc3/data3/jake_dp03/for_mario/mpcorb_eph_*.hdf -j 24 --output today.mpsky.bin
```
where the .hdf files are outputs of Sorcha for a single night.

To query it directly from the resulting file, run:
```
mpsky query --source today.mpsky.bin 60853.1 32 11 --radius=1.8
```
To serve the cache via a HTTP endpoint:
```
mpsky serve --verbose
```
this will bind to localhost:8000 by default.

To query from the HTTP endpoint:
```
mpsky query 60853.1 32 11 --radius=1.8 --source http://localhost:8000/ephemerides
```

## How to develop

TBD.
