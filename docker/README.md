# Dockerfile for the mpsky service

WARNING: this is not yet ready for broad use. best not to try it at all.

Note (FIXME): right now we built a mpsky cache file into the image; this
will go away in the near future. These images aren't stored in the github
repo, but should be rsynced from SLAC:

```
rsync -avz slacd:projects/github.com/mjuric/lsst-gen-ephemcache/outputs/caches .
```

FIXME: This Dockerfile is irreproducible as it doesn't record any package
versions.

## Build Image

```
$ make build
```

## Push to ghcr repository

```
$ make push
```

## Test

Run with files from the built-in cache dir:

```
docker run -it --rm --tmpfs /tmp:size=100m -p 8000:8000 --read-only mpsky-daily --host 0.0.0.0 --verbose
```

or mount a different dir

```
docker run -it --rm --tmpfs /tmp:size=100m -p 8000:8000 --read-only -v ./caches:/caches mpsky-daily --host 0.0.0.0 --verbose
```

To test that it works, run:
```
mpsky query 60792.8 32 11 --radius=1.8 --source http://localhost:8000/ephemerides
```
