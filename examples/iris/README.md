# Iris

Run from the repository root.

```sh
# Download
curl -fL https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -o iris.data

# Run
cabal run iris-example -- train iris.data iris.safetensors
cabal run iris-example -- evaluate iris.data iris.safetensors
```
