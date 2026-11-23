# MNIST

Run from the repository root.

```sh
# Download
curl -fL --create-dirs -o 'mnist/#1-#2-ubyte.gz' \
    'https://ossci-datasets.s3.amazonaws.com/mnist/{train,t10k}-{images-idx3,labels-idx1}-ubyte.gz'
gzip -d mnist/*.gz

# Run
cabal run mnist-example -- train mnist mnist.safetensors
cabal run mnist-example -- evaluate mnist mnist.safetensors
```
