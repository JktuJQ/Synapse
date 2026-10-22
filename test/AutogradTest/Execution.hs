{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module AutogradTest.Execution (tests) where

import qualified AutogradTest.Execution.Public
import GHC.Generics (Generic)
import qualified Hedgehog as H
import qualified Numeric.Backprop.Explicit as BP
import qualified Synapse.Autograd as AG
import qualified Synapse.Autograd.Differentiable as D
import qualified Synapse.Tensor as T
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

newtype Distance = Distance Double deriving (Eq, Show, Num)

instance AG.Backpropagatable Distance

newtype DistanceVar s = DistanceVar (AG.Var s Distance)

instance AG.Differentiable Distance where
  type WithGrad s Distance = DistanceVar s
  structure = D.leaf DistanceVar (\(DistanceVar value) -> value)

data Point = Point Double Double deriving (Eq, Show)

instance AG.Backpropagatable Point where
  zero _ = Point 0 0
  one _ = Point 1 1
  add (Point x y) (Point u v) = Point (x + u) (y + v)

newtype PointVar s = PointVar (AG.Var s Point)

instance AG.Differentiable Point where
  type WithGrad s Point = PointVar s
  structure = D.leaf PointVar (\(PointVar value) -> value)

projection :: (AG.Context s) => PointVar s -> AG.Var s Double
projection (PointVar (AG.Var value)) =
  AG.Var $
    BP.liftOp1
      (BP.AF AG.add)
      (BP.op1 (\(Point x y) -> (x + 2 * y, \seed -> Point seed (2 * seed))))
      value

data Parameters a b = Parameters a b deriving (Eq, Show, Generic)

instance (AG.Differentiable a, AG.Differentiable b) => AG.Differentiable (Parameters a b) where
  type WithGrad s (Parameters a b) = Parameters (AG.WithGrad s a) (AG.WithGrad s b)

newtype Wrapped a = Wrapped a deriving (Eq, Show, Generic)

instance (AG.Differentiable a) => AG.Differentiable (Wrapped a) where
  type WithGrad s (Wrapped a) = Wrapped (AG.WithGrad s a)

data Manual a b = Manual {manualFirst :: a, manualSecond :: b} deriving (Eq, Show)

instance (AG.Differentiable a, AG.Differentiable b) => AG.Differentiable (Manual a b) where
  type WithGrad s (Manual a b) = Manual (AG.WithGrad s a) (AG.WithGrad s b)
  structure =
    D.record Manual Manual
      D.<+> D.field manualFirst manualFirst
      D.<+> D.field manualSecond manualSecond

data Wide a = Wide a a a a a a a a a a a a deriving (Eq, Show, Generic)

instance (AG.Differentiable a) => AG.Differentiable (Wide a) where
  type WithGrad s (Wide a) = Wide (AG.WithGrad s a)

data DenseParameters t = DenseParameters
  { denseWeights :: t '[2, 3] Double,
    denseBias :: t '[3] Double
  }
  deriving (Generic)

instance AG.Differentiable (DenseParameters T.Tensor) where
  type WithGrad s (DenseParameters T.Tensor) = DenseParameters (AG.Tensor s)

tensorValues :: (T.Shape sh) => T.Tensor sh Double -> [Double]
tensorValues = T.toList . T.run T.CPU

tests :: TestTree
tests =
  testGroup
    "execution"
    [ testProperty "non-numeric leaves initialize gradients without a Num instance" $ H.withTests 1 $ H.property $ do
        let input = Point 3 4
        AG.backprop id input H.=== (input, Point 1 1)
        AG.backprop (const (AG.constant (7 :: Double))) input H.=== (7, Point 0 0),
      testProperty "non-numeric leaf gradients accumulate across multiple uses" $ H.withTests 1 $ H.property $ do
        AG.backprop (\point -> projection point * projection point) (Point 3 4) H.=== (121, Point 22 44)
        let (value, backward) = AG.pullback projection (Point 3 4)
        value H.=== 11
        backward 3 H.=== Point 3 6
        backward 5 H.=== Point 5 10,
      testProperty "non-numeric leaves support constants and stopped history" $ H.withTests 1 $ H.property $ do
        let input = Point 3 4
        AG.evaluate (const (AG.constant input)) (0 :: Double) H.=== input
        AG.backprop AG.stopGrad input H.=== (input, Point 0 0),
      testProperty "ordinary scalar constants participate in values but not input gradients" $ H.withTests 1 $ H.property $ do
        AG.backprop (\x -> x * AG.constant (3 :: Double)) (2 :: Double) H.=== (6, 3)
        AG.backprop (const (AG.constant (7 :: Double))) (2 :: Double) H.=== (7, 0)
        AG.evaluate (\x -> x + AG.constant (4 :: Double)) (2 :: Double) H.=== 6,
      testProperty "nested Generic constants preserve heterogeneous fields" $ H.withTests 1 $ H.property $ do
        let frozen = Wrapped (Parameters (3 :: Double) (4 :: Float))
            objective x = let Wrapped (Parameters a b) = AG.constant frozen in (x * a, b * b)
        AG.backprop objective (2 :: Double) H.=== ((6, 16), 3),
      testProperty "a frozen tensor backbone supplies constant features to a trainable head" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3, 4] :: T.Tensor '[2, 2] Double
            frozenWeights = T.unsafeFromList [2, 0, 0, 3] :: T.Tensor '[2, 2] Double
            features = T.matmul input frozenWeights
            headWeights = T.unsafeFromList [1, 2] :: T.Tensor '[2, 1] Double
            (value, gradient) = AG.backprop (T.sum . T.matmul (AG.constant features)) headWeights
        tensorValues value H.=== [44]
        tensorValues gradient H.=== [8, 18],
      testProperty "frozen tensor weights still allow gradients with respect to their input" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3, 4] :: T.Tensor '[2, 2] Double
            frozenWeights = T.unsafeFromList [2, 0, 0, 3] :: T.Tensor '[2, 2] Double
            gradient = AG.gradient (\x -> T.sum (T.matmul x (AG.constant frozenWeights))) input
        tensorValues gradient H.=== [2, 3, 2, 3],
      testProperty "constant outputs return shape-correct zeros for unused tensor inputs" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
        tensorValues (AG.gradient (const (AG.constant (7 :: Double))) input) H.=== [0, 0, 0],
      testProperty "Execution exposes constants without requiring a context or wrapper constructors" $ H.withTests 1 $ H.property $ do
        let frozen = T.unsafeFromList [2, 3, 5] :: T.Tensor '[3] Double
        tensorValues (AutogradTest.Execution.Public.constantGradient frozen frozen) H.=== [2, 3, 5],
      testProperty "element expressions can be introduced as constants" $ H.withTests 1 $ H.property $ do
        let (value, gradient) = AG.backprop (\x -> x * AG.constant (T.constant (3 :: Double))) (T.constant (2 :: Double))
        tensorValues (T.scalar value) H.=== [6]
        tensorValues (T.scalar gradient) H.=== [3],
      testProperty "Generic records preserve mixed and nested field gradients" $ H.withTests 1 $ H.property $ do
        let input = Wrapped (Parameters (3 :: Double) (4 :: Float))
            objective (Wrapped (Parameters x y)) = Parameters (x * x) (y * y)
            (value, backward) = AG.pullback objective input
        value H.=== Parameters 9 16
        backward (Parameters 2 3) H.=== Wrapped (Parameters 12 24)
        AG.backprop objective input H.=== (Parameters 9 16, Wrapped (Parameters 6 8))
        AG.evaluate objective input H.=== Parameters 9 16,
      testProperty "Generic records support more than nine fields" $ H.withTests 1 $ H.property $ do
        let input = Wide 1 2 3 4 5 6 7 8 9 10 11 12 :: Wide Double
            (value, gradient) = AG.backprop id input
        value H.=== input
        gradient H.=== Wide 1 1 1 1 1 1 1 1 1 1 1 1,
      testProperty "Generic record stopGrad cuts each field history" $ H.withTests 1 $ H.property $ do
        let input = Parameters (3 :: Double) (4 :: Double)
            objective parameters = let Parameters x y = AG.stopGrad parameters in x * x + y
        AG.gradient objective input H.=== Parameters 0 0,
      testProperty "manual structures support records without Generic" $ H.withTests 1 $ H.property $ do
        let input = Manual (3 :: Double) (4 :: Float)
            objective (Manual x y) = Manual (x * x) (y * y)
        AG.backprop objective input H.=== (Manual 9 16, Manual 6 8)
        AG.evaluate objective input H.=== Manual 9 16
        AG.gradient (AG.stopGrad . objective) input H.=== Manual 0 0,
      testProperty "Generic tensor parameters register fields with different shapes" $ H.withTests 1 $ H.property $ do
        let input = DenseParameters (T.unsafeFromList [1 .. 6]) (T.unsafeFromList [7, 8, 9])
            objective (DenseParameters w b) = T.sum (w * w + T.broadcast b)
            (value, DenseParameters dw db) = AG.backprop objective input
        tensorValues value H.=== [139]
        tensorValues dw H.=== [2, 4, 6, 8, 10, 12]
        tensorValues db H.=== [2, 2, 2],
      testProperty "scalar entry points infer their representations" $ H.withTests 1 $ H.property $ do
        AG.evaluate (\x -> x * x + x) (3 :: Double) H.=== 12
        AG.backprop (\x -> x * x + x) (3 :: Double) H.=== (12, 7)
        AG.gradient (\x -> x * x + x) (3 :: Double) H.=== 7,
      testProperty "pullbacks accept different output seeds repeatedly" $ H.withTests 1 $ H.property $ do
        let (value, backward) = AG.pullback (\x -> x * x) (3 :: Double)
        value H.=== 9
        backward 1 H.=== 6
        backward 4 H.=== 24
        backward 0 H.=== 0
        backward 1 H.=== 6,
      testProperty "constant and unused inputs receive zero gradients" $ H.withTests 1 $ H.property $ do
        AG.backprop (const 7) (3 :: Double) H.=== (7 :: Double, 0)
        AG.gradient (\(x, _) -> x * x) (3 :: Double, 9 :: Float) H.=== (6, 0),
      testProperty "nested heterogeneous inputs and outputs preserve their structure" $ H.withTests 1 $ H.property $ do
        let (value, backward) = AG.pullback (\(x, (y, _)) -> (x * x, y * y)) (3 :: Double, (4 :: Float, 7 :: Double))
        value H.=== (9, 16)
        backward (2, 3) H.=== (12, (24, 0))
        AG.backprop (\(x, y) -> (x * x, y * y)) (3 :: Double, 4 :: Float) H.=== ((9, 16), (6, 8)),
      testProperty "nested layer parameters have no nine-component limit" $ H.withTests 1 $ H.property $ do
        let layer = (2, 3) :: (Double, Double)
            parameters = (layer, (layer, (layer, (layer, (layer, (layer, (layer, (layer, (layer, (layer, (layer, layer)))))))))))
            objective (a, (b, (c, (d, (e, (f, (g, (h, (i, (j, (k, l))))))))))) = sum [weight * bias | (weight, bias) <- [a, b, c, d, e, f, g, h, i, j, k, l]]
        AG.gradient objective parameters H.=== ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), ((3, 2), (3, 2)))))))))))),
      testProperty "tuples up to nine components are accepted as inputs and outputs" $ H.withTests 1 $ H.property $ do
        let input = (1, 2, 3, 4, 5, 6, 7, 8, 9) :: (Double, Double, Double, Double, Double, Double, Double, Double, Double)
            (value, gradient) = AG.backprop id input
        value H.=== input
        gradient H.=== (1, 1, 1, 1, 1, 1, 1, 1, 1),
      testProperty "custom numeric types and variable newtypes need no runner adapter" $ H.withTests 1 $ H.property $ do
        let result = AG.backprop (\(DistanceVar x) -> DistanceVar (x * x)) (Distance 3)
        result H.=== (Distance 9, Distance 6),
      testProperty "stopGrad preserves values and cuts the stopped contribution" $ H.withTests 1 $ H.property $ do
        AG.backprop (\x -> x * AG.stopGrad x) (3 :: Double) H.=== (9, 3)
        AG.gradient (\x -> AG.stopGrad (x * x)) (3 :: Double) H.=== 0,
      testProperty "stopGrad prevents singular upstream derivatives from being evaluated" $ H.withTests 1 $ H.property $ do
        AG.gradient (\x -> AG.stopGrad (sqrt x) + x) (0 :: Double) H.=== 1
        AG.gradient (AG.stopGrad . sqrt) (0 :: Double) H.=== 0,
      testProperty "evaluate and stopGrad do not request an unavailable pullback" $ H.withTests 1 $ H.property $ do
        let function x = AG.Var (BP.liftOp1 BP.afNum (BP.noGrad1 (+ 1)) (AG.unVar x))
        AG.evaluate function (3 :: Double) H.=== 4
        AG.gradient (AG.stopGrad . function) (3 :: Double) H.=== 0,
      testProperty "stopped branches cannot replace the requested tape output" $ H.withTests 1 $ H.property $ do
        let function x = let result = x * x in result `seq` AG.stopGrad (x * x * x) `seq` result
        AG.backprop function (3 :: Double) H.=== (9, 6),
      testProperty "stopGrad works on nested structures" $ H.withTests 1 $ H.property $ do
        AG.gradient (\pair -> let (x, y) = AG.stopGrad pair in x * x + y) (3 :: Double, 4 :: Double) H.=== (0, 0),
      testProperty "tensor results remain staged and default seeds sum their elements" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
            (value, gradient) = AG.backprop (\x -> x * x) input
        tensorValues value H.=== [1, 4, 9]
        tensorValues gradient H.=== [2, 4, 6]
        tensorValues (AG.evaluate (\x -> x * x) input) H.=== [1, 4, 9],
      testProperty "Execution works without importing tensor wrapper constructors" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
        tensorValues (AutogradTest.Execution.Public.squaredGradient input) H.=== [2, 4, 6],
      testProperty "tensor pullbacks accept nonuniform seeds" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
            (_, backward) = AG.pullback (\x -> x * x) input
        tensorValues (backward (T.unsafeFromList [2, 3, 5])) H.=== [4, 12, 30]
        tensorValues (backward (T.unsafeFromList [1, 1, 1])) H.=== [2, 4, 6],
      testProperty "parameters of different tensor shapes have separate gradients" $ H.withTests 1 $ H.property $ do
        let weights = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            bias = T.unsafeFromList [7, 8, 9] :: T.Tensor '[3] Double
            (value, (dw, db)) = AG.backprop (\(w, b) -> T.sum (w * w + T.broadcast b)) (weights, bias)
        tensorValues value H.=== [139]
        tensorValues dw H.=== [2, 4, 6, 8, 10, 12]
        tensorValues db H.=== [2, 2, 2],
      testProperty "unused tensor parameters receive zeros of the original shape" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2] :: T.Tensor '[2] Double
            unused = T.unsafeFromList [1 .. 6] :: T.Tensor '[2, 3] Double
            (_, gradient) = AG.gradient (\(x, _) -> T.sum x) (input, unused)
        tensorValues gradient H.=== replicate 6 0,
      testProperty "tensor stopGrad removes the history while preserving the staged value" $ H.withTests 1 $ H.property $ do
        let input = T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double
            (value, gradient) = AG.backprop (\x -> T.sum (x * AG.stopGrad x)) input
        tensorValues value H.=== [14]
        tensorValues gradient H.=== [1, 2, 3],
      testProperty "element expressions use the same entry points" $ H.withTests 1 $ H.property $ do
        let (value, gradient) = AG.backprop (\x -> x * x) (T.constant (3 :: Double))
        tensorValues (T.scalar value) H.=== [9]
        tensorValues (T.scalar gradient) H.=== [6],
      testProperty "autograd can run inside a prepared tensor function" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU (AG.backprop (\x -> T.sum (x * x)))
            first = T.run T.CPU (T.unsafeFromList [1, 2, 3] :: T.Tensor '[3] Double)
            second = T.run T.CPU (T.unsafeFromList [4, 5, 6] :: T.Tensor '[3] Double)
            (firstValue, firstGradient) = execute first
            (secondValue, secondGradient) = execute second
        T.toList firstValue H.=== [14]
        T.toList firstGradient H.=== [2, 4, 6]
        T.toList secondValue H.=== [77]
        T.toList secondGradient H.=== [8, 10, 12]
    ]
