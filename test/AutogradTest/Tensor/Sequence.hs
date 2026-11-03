{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module AutogradTest.Tensor.Sequence (tests) where

import Control.Monad.Trans.State.Strict (State, runState, state)
import qualified Data.Array.Accelerate as A
import Data.Proxy (Proxy (..))
import GHC.Generics (Generic)
import GHC.TypeNats (KnownNat, natVal)
import qualified Hedgehog as H
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T
import qualified Synapse.Tensor.Ops as Ops
import qualified Synapse.Tensor.Type as Raw
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

data Parameters (t :: T.TensorLike) = Parameters (t '[] Double) (t '[] Double) deriving (Generic)

type instance T.WithTensor t (Parameters T.Tensor) = Parameters t

data Carry (t :: T.TensorLike) = Carry (t '[] Double) (t '[] Double) deriving (Generic)

type instance T.WithTensor t (Carry T.Tensor) = Carry t

type Input n t = (Parameters t, (t '[n] Double, Carry t))

type Output n t = (t '[n] Double, Carry t)

step :: (Floating (t '[] Double)) => Parameters t -> t '[] Double -> State (Carry t) (t '[] Double)
step (Parameters weight bias) x = state $ \(Carry h c) ->
  let nextC = 0.7 * c + x * weight
      nextH = tanh $ nextC + h * bias
   in (nextH + nextC, Carry nextH nextC)

compact :: forall n t. (KnownNat n, T.TensorOps t Double) => Input n t -> Output n t
compact (p, (xs, initial)) = runState (T.mapAccum step p xs) initial

-- The reference expands host iteration into ordinary differentiable operations.
readItem :: forall n t. (KnownNat n, T.TensorOps t Double) => Int -> t '[n] Double -> t '[] Double
readItem i =
  Ops.customVJP
    Ops.Numeric
    (\(Raw.Tensor xs) -> Raw.Tensor $ A.unit $ xs A.! A.index1 (A.constant i))
    (\(Raw.Tensor xs) -> (Raw.Tensor $ A.unit $ xs A.! A.index1 (A.constant i), ()))
    (\() (Raw.Tensor seed) -> Raw.Tensor $ A.generate (A.constant $ A.Z A.:. count) $ \ix -> A.cond (A.indexHead ix A.== A.constant i) (A.the seed) 0)
  where
    count = fromIntegral $ natVal $ Proxy @n

writeItem :: forall n t. (KnownNat n, T.TensorOps t Double) => Int -> t '[] Double -> t '[n] Double -> t '[n] Double
writeItem i value target =
  Ops.customVJP
    Ops.Numeric
    (\(Raw.Tensor x, Raw.Tensor xs) -> Raw.Tensor $ A.generate (A.shape xs) $ \ix -> A.cond (A.indexHead ix A.== A.constant i) (A.the x) (xs A.! ix))
    (\(Raw.Tensor x, Raw.Tensor xs) -> (Raw.Tensor $ A.generate (A.shape xs) $ \ix -> A.cond (A.indexHead ix A.== A.constant i) (A.the x) (xs A.! ix), ()))
    (\() (Raw.Tensor seed) -> (Raw.Tensor $ A.unit $ seed A.! A.index1 (A.constant i), Raw.Tensor $ A.generate (A.shape seed) $ \ix -> A.cond (A.indexHead ix A.== A.constant i) 0 (seed A.! ix)))
    (value, target)

unrolled :: forall n t. (KnownNat n, T.TensorOps t Double, Floating (t '[] Double), Num (t '[n] Double)) => Input n t -> Output n t
unrolled (p, (xs, initial)) =
  let count = fromIntegral $ natVal $ Proxy @n
      (ys, final) = runState (traverse (step p . (`readItem` xs)) [0 .. count - 1]) initial
   in (foldl (\acc (i, y) -> writeItem i y acc) 0 $ zip [0 ..] ys, final)

outputValues :: (KnownNat n) => T.WithTensor T.TensorData (Output n T.Tensor) -> [Double]
outputValues (ys, Carry h c) = T.toList ys ++ T.toList h ++ T.toList c

inputValues :: (KnownNat n) => T.WithTensor T.TensorData (Input n T.Tensor) -> [Double]
inputValues (Parameters w b, (xs, Carry h c)) = concat [T.toList w, T.toList b, T.toList xs, T.toList h, T.toList c]

checkLength :: forall n. (KnownNat n) => H.PropertyT IO ()
checkLength = do
  let count :: Int
      count = fromIntegral $ natVal $ Proxy @n
      sample offset = (Parameters 0.3 (-0.2), (T.fromList [sin (fromIntegral i * 0.1) + offset | i <- [1 .. count]], Carry 0.1 (-0.1))) :: Input n T.Tensor
      seed = (T.fromList [cos (fromIntegral i * 0.3) | i <- [1 .. count]], Carry 0.3 (-0.7)) :: Output n T.Tensor
      compiled = T.run1 T.CPU $ \input -> let (output, backward) = AG.pullback (compact @n) input in (output, backward seed)
      reference = T.run1 T.CPU $ \input -> let (output, backward) = AG.pullback (unrolled @n) input in (output, backward seed)
      inference = T.run1 T.CPU $ compact @n
  mapM_
    ( \offset -> do
        let input = T.run T.CPU $ sample offset
            (actual, gradient) = compiled input
            (expected, expectedGradient) = reference input
        approxList (outputValues actual) $ outputValues expected
        approxList (inputValues gradient) $ inputValues expectedGradient
        approxList (outputValues $ inference input) $ outputValues expected
    )
    [0, 0.05]

type MatrixInput t = ((t '[2, 3] Double, t '[3, 3] Double, t '[3] Double), (t '[4, 2, 2] Double, t '[2, 3] Double))

type MatrixOutput t = (t '[4, 2, 3] Double, t '[2, 3] Double)

matrixStep :: (T.TensorOps t Double, Floating (t '[2, 3] Double)) => (t '[2, 3] Double, t '[3, 3] Double, t '[3] Double) -> t '[2, 2] Double -> State (t '[2, 3] Double) (t '[2, 3] Double)
matrixStep (weights, recurrent, bias) x = state $ \h ->
  let next = tanh $ T.matmul x weights + T.matmul h recurrent + T.broadcast bias
   in (next, next)

matrixScan :: (T.TensorOps t Double) => MatrixInput t -> MatrixOutput t
matrixScan (p, (xs, initial)) = runState (T.mapAccum matrixStep p xs) initial

matrixReference :: (T.TensorOps t Double, Floating (t '[2, 3] Double)) => MatrixInput t -> MatrixOutput t
matrixReference (p, (xs, initial)) =
  let (y0, s0) = runState (matrixStep p $ T.slice @'[ 'T.At 0, 'T.All, 'T.All] xs) initial
      (y1, s1) = runState (matrixStep p $ T.slice @'[ 'T.At 1, 'T.All, 'T.All] xs) s0
      (y2, s2) = runState (matrixStep p $ T.slice @'[ 'T.At 2, 'T.All, 'T.All] xs) s1
      (y3, s3) = runState (matrixStep p $ T.slice @'[ 'T.At 3, 'T.All, 'T.All] xs) s2
   in (T.concat @0 (T.stack @0 y0 y1) (T.stack @0 y2 y3), s3)

matrixOutputValues :: T.WithTensor T.TensorData (MatrixOutput T.Tensor) -> [Double]
matrixOutputValues (ys, h) = T.toList ys ++ T.toList h

matrixInputValues :: T.WithTensor T.TensorData (MatrixInput T.Tensor) -> [Double]
matrixInputValues ((w, r, b), (xs, h)) = concat [T.toList w, T.toList r, T.toList b, T.toList xs, T.toList h]

checkMatrix :: H.PropertyT IO ()
checkMatrix = do
  let input :: MatrixInput T.Tensor
      input = ((T.fromList [0.1, -0.2, 0.3, 0.05, 0.2, -0.1], T.fromList $ replicate 9 0.02, T.fromList [0.01, -0.02, 0.03]), (T.fromList [sin $ fromIntegral i * 0.1 | i <- [1 .. 16 :: Int]], T.full 0.1))
      seed :: MatrixOutput T.Tensor
      seed = (T.fromList [cos $ fromIntegral i * 0.2 | i <- [1 .. 24 :: Int]], T.fromList [0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
      execute = T.run1 T.CPU $ \arguments -> let (output, backward) = AG.pullback matrixScan arguments in (output, backward seed)
      reference = T.run1 T.CPU $ \arguments -> let (output, backward) = AG.pullback matrixReference arguments in (output, backward seed)
      material = T.run T.CPU input
      (actual, gradient) = execute material
      (expected, expectedGradient) = reference material
  approxList (matrixOutputValues actual) $ matrixOutputValues expected
  approxList (matrixInputValues gradient) $ matrixInputValues expectedGradient
  approxList (matrixOutputValues $ T.run T.CPU $ matrixScan input) $ matrixOutputValues expected

tests :: TestTree
tests =
  testGroup
    "sequence traversal"
    [ testProperty "batched matrix steps and multidimensional histories" $ H.withTests 1 $ H.property checkMatrix,
      testProperty "empty scan preserves initial state and its cotangent" $ H.withTests 1 $ H.property $ checkLength @0,
      testProperty "one step matches an ordinary State computation" $ H.withTests 1 $ H.property $ checkLength @1,
      testProperty "all outputs, parameters, inputs and initial state gradients" $ H.withTests 1 $ H.property $ checkLength @4,
      testProperty "prepared scan accepts changing runtime inputs" $ H.withTests 1 $ H.property $ checkLength @16,
      testProperty "longer scan agrees with expanded differentiation" $ H.withTests 1 $ H.property $ checkLength @64
    ]
