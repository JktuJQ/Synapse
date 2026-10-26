{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE TypeApplications #-}

module AutogradTest.Tensor.Lifting (tests) where

import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Ops as Ops
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

values :: (T.Shape sh) => T.Tensor sh Double -> [Double]
values = T.toList . T.run T.CPU

weighted1 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted1 =
  Ops.lift1
    id
    (\_ _ gradient -> gradient)

weighted2 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted2 input =
  Ops.lift2
    (\x1 x2 -> x1 + 2 * x2)
    (\_ _ _ gradient -> (gradient, 2 * gradient))
    input
    input

weighted3 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted3 input =
  Ops.lift3
    (\x1 x2 x3 -> x1 + 2 * x2 + 3 * x3)
    (\_ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient))
    input
    input
    input

weighted4 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted4 input =
  Ops.lift4
    (\x1 x2 x3 x4 -> x1 + 2 * x2 + 3 * x3 + 4 * x4)
    (\_ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient))
    input
    input
    input
    input

weighted5 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted5 input =
  Ops.lift5
    (\x1 x2 x3 x4 x5 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5)
    (\_ _ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient, 5 * gradient))
    input
    input
    input
    input
    input

weighted6 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted6 input =
  Ops.lift6
    (\x1 x2 x3 x4 x5 x6 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6)
    (\_ _ _ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient, 5 * gradient, 6 * gradient))
    input
    input
    input
    input
    input
    input

weighted7 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted7 input =
  Ops.lift7
    (\x1 x2 x3 x4 x5 x6 x7 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7)
    (\_ _ _ _ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient, 5 * gradient, 6 * gradient, 7 * gradient))
    input
    input
    input
    input
    input
    input
    input

weighted8 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted8 input =
  Ops.lift8
    (\x1 x2 x3 x4 x5 x6 x7 x8 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7 + 8 * x8)
    (\_ _ _ _ _ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient, 5 * gradient, 6 * gradient, 7 * gradient, 8 * gradient))
    input
    input
    input
    input
    input
    input
    input
    input

weighted9 :: (T.TensorOps t Double) => t '[] Double -> t '[] Double
weighted9 input =
  Ops.lift9
    (\x1 x2 x3 x4 x5 x6 x7 x8 x9 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7 + 8 * x8 + 9 * x9)
    (\_ _ _ _ _ _ _ _ _ _ gradient -> (gradient, 2 * gradient, 3 * gradient, 4 * gradient, 5 * gradient, 6 * gradient, 7 * gradient, 8 * gradient, 9 * gradient))
    input
    input
    input
    input
    input
    input
    input
    input
    input

mixedShapes :: (T.TensorOps t Double) => t '[2] Double -> t '[1, 1] Double -> t '[] Double -> t '[] Double
mixedShapes =
  Ops.lift3
    (\x y z -> T.sum x + 2 * T.sum y + 3 * z)
    (\_ _ _ _ gradient -> (T.broadcast gradient, T.broadcast (2 * gradient), 3 * gradient))

expWithOutput :: (T.TensorOps t Double) => t '[2] Double -> t '[2] Double
expWithOutput = Ops.lift1 exp (\_ output gradient -> output * gradient)

type ScalarInputs (t :: T.TensorLike) = (t '[] Double, t '[] Double, t '[] Double, t '[] Double, t '[] Double, t '[] Double, t '[] Double, t '[] Double, t '[] Double)

weightedInputs9 ::
  (T.TensorOps t Double) =>
  ScalarInputs t ->
  t '[] Double
weightedInputs9 (a, b, c, d, e, f, g, h, i) =
  Ops.lift9
    (\u v w x y z p q r -> u + 2 * v + 3 * w + 4 * x + 5 * y + 6 * z + 7 * p + 8 * q + 9 * r)
    (\_ _ _ _ _ _ _ _ _ _ dy -> (dy, 2 * dy, 3 * dy, 4 * dy, 5 * dy, 6 * dy, 7 * dy, 8 * dy, 9 * dy))
    a
    b
    c
    d
    e
    f
    g
    h
    i

tests :: TestTree
tests =
  testGroup
    "tensor primitive lifting"
    [ testProperty "lift1 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted1 input
        values (weighted1 input) H.=== [2]
        values output H.=== [2]
        values (backward (T.full @'[] 7)) H.=== [7],
      testProperty "ordinary lift1 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift1 id (error "inference forced a pullback") input
        values output H.=== [2],
      testProperty "lift2 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted2 input
        values (weighted2 input) H.=== [6]
        values output H.=== [6]
        values (backward (T.full @'[] 7)) H.=== [21],
      testProperty "ordinary lift2 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift2 (\x1 x2 -> x1 + 2 * x2) (error "inference forced a pullback") input input
        values output H.=== [6],
      testProperty "lift3 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted3 input
        values (weighted3 input) H.=== [12]
        values output H.=== [12]
        values (backward (T.full @'[] 7)) H.=== [42],
      testProperty "ordinary lift3 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift3 (\x1 x2 x3 -> x1 + 2 * x2 + 3 * x3) (error "inference forced a pullback") input input input
        values output H.=== [12],
      testProperty "lift4 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted4 input
        values (weighted4 input) H.=== [20]
        values output H.=== [20]
        values (backward (T.full @'[] 7)) H.=== [70],
      testProperty "ordinary lift4 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift4 (\x1 x2 x3 x4 -> x1 + 2 * x2 + 3 * x3 + 4 * x4) (error "inference forced a pullback") input input input input
        values output H.=== [20],
      testProperty "lift5 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted5 input
        values (weighted5 input) H.=== [30]
        values output H.=== [30]
        values (backward (T.full @'[] 7)) H.=== [105],
      testProperty "ordinary lift5 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift5 (\x1 x2 x3 x4 x5 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5) (error "inference forced a pullback") input input input input input
        values output H.=== [30],
      testProperty "lift6 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted6 input
        values (weighted6 input) H.=== [42]
        values output H.=== [42]
        values (backward (T.full @'[] 7)) H.=== [147],
      testProperty "ordinary lift6 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift6 (\x1 x2 x3 x4 x5 x6 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6) (error "inference forced a pullback") input input input input input input
        values output H.=== [42],
      testProperty "lift7 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted7 input
        values (weighted7 input) H.=== [56]
        values output H.=== [56]
        values (backward (T.full @'[] 7)) H.=== [196],
      testProperty "ordinary lift7 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift7 (\x1 x2 x3 x4 x5 x6 x7 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7) (error "inference forced a pullback") input input input input input input input
        values output H.=== [56],
      testProperty "lift8 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted8 input
        values (weighted8 input) H.=== [72]
        values output H.=== [72]
        values (backward (T.full @'[] 7)) H.=== [252],
      testProperty "ordinary lift8 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift8 (\x1 x2 x3 x4 x5 x6 x7 x8 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7 + 8 * x8) (error "inference forced a pullback") input input input input input input input input
        values output H.=== [72],
      testProperty "lift9 accumulates all input edges and respects an output seed" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            (output, backward) = AG.pullback weighted9 input
        values (weighted9 input) H.=== [90]
        values output H.=== [90]
        values (backward (T.full @'[] 7)) H.=== [315],
      testProperty "ordinary lift9 does not evaluate its pullback" $ H.withTests 1 $ H.property $ do
        let input = T.full @'[] (2 :: Double)
            output = Ops.lift9 (\x1 x2 x3 x4 x5 x6 x7 x8 x9 -> x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5 + 6 * x6 + 7 * x7 + 8 * x8 + 9 * x9) (error "inference forced a pullback") input input input input input input input input input
        values output H.=== [90],
      testProperty "lift9 preserves the order of independently registered inputs" $ H.withTests 1 $ H.property $ do
        let input n = T.full @'[] (n :: Double)
            (output, backward) =
              AG.pullback @(ScalarInputs T.Tensor) @(T.Tensor '[] Double)
                weightedInputs9
                (input 1, input 2, input 3, input 4, input 5, input 6, input 7, input 8, input 9)
            (da, db, dc, dd, de, df, dg, dh, di) = backward (input 7)
        values output H.=== [285]
        concatMap values [da, db, dc, dd, de, df, dg, dh, di] H.=== [7, 14, 21, 28, 35, 42, 49, 56, 63],
      testProperty "input shapes and gradient results remain independent" $ H.withTests 1 $ H.property $ do
        let inputs = (T.fromList @'[2] [1, 2 :: Double], T.full @'[1, 1] 4, T.full @'[] 5)
            (output, (dx, dy, dz)) = AG.backprop (\(x, y, z) -> mixedShapes x y z) inputs
        values output H.=== [26]
        values dx H.=== [1, 1]
        values dy H.=== [2]
        values dz H.=== [3],
      testProperty "pullbacks can reuse the forward output" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [0, 1 :: Double]
            (output, gradient) = AG.backprop expWithOutput input
        approxList (values output) [1, exp 1]
        approxList (values gradient) [1, exp 1],
      testProperty "shape operations on boolean tensors do not require numeric instances" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [False, True]
        T.toList (T.run T.CPU (T.broadcast @'[2, 2] input)) H.=== [False, True, False, True]
        T.toList (T.run T.CPU (T.transpose (T.reshape @'[1, 2] input))) H.=== [False, True],
      testProperty "literal and primal preserve values without preserving incoming gradients" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [2, 3 :: Double]
            (output, gradient) = AG.backprop (\x -> T.sum (x * Ops.literal (Ops.primal x))) input
        values output H.=== [13]
        values gradient H.=== [2, 3],
      testProperty "plain lift1 accepts a boolean input without using its numeric pullback" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2] [False, True]
            output = Ops.lift1 id (error "inference forced a numeric pullback") input
        T.toList (T.run T.CPU output) H.=== [False, True]
    ]
