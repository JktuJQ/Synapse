<div align="center">
  <h1><img src="SynapseLogo.png" width="160" alt="Synapse logo"><br>Synapse</h1>
</div>

`Synapse` is a machine learning library for Haskell.
It provides tensors, automatic differentiation, and tools for building and training neural networks.

## 🔨 Design 🔨

Haskell has a handful of neural network libraries, each with its own approach.
The [`neural`][neural] library composes models and uses pipes to assemble training loops.
[Grenade][grenade] uses the type system to check that layers fit together.
[Hasktorch][hasktorch] builds on LibTorch and brings its tensors and neural network operations to Haskell.

`Synapse` brings those ideas together, while still remaining easily extensible, simple and powerful:

* `Arrow`-based composition
* Type-level shapes and checks
* Backend-accelerated execution

**Interested?** Let's dive right in!

## 💻 Usage 💻


[neural]: https://hackage.haskell.org/package/neural
[grenade]: https://hackage.haskell.org/package/grenade
[hasktorch]: https://hackage.haskell.org/package/hasktorch
