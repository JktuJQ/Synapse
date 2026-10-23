{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MonoLocalBinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

module Main (main) where

import GHC.Generics (Generic)
import qualified Synapse.Autograd as AG
import qualified Synapse.Tensor as T

data Parameters t = Parameters
  { weights :: t '[2, 1] Double,
    bias :: t '[1] Double
  }
  deriving (Generic)

instance AG.Differentiable (Parameters T.Tensor) where
  type WithGrad s (Parameters T.Tensor) = Parameters (AG.Tensor s)

predict :: (AG.Context s) => Parameters (AG.Tensor s) -> AG.Tensor s '[4, 2] Double -> AG.Tensor s '[4, 1] Double
predict parameters input = T.matmul input (weights parameters) + T.broadcast (bias parameters)

objective :: (AG.Context s) => T.Tensor '[4, 2] Double -> T.Tensor '[4, 1] Double -> Parameters (AG.Tensor s) -> AG.Tensor s '[] Double
objective input target parameters =
  let residual = predict parameters (AG.constant input) - AG.constant target
   in T.mean (residual * residual)

showTensor :: (T.Shape sh) => String -> T.Tensor sh Double -> IO ()
showTensor label = showData label . T.run T.CPU

showData :: forall sh. (T.Shape sh) => String -> T.TensorData sh Double -> IO ()
showData label tensor = putStrLn (label ++ " " ++ show (T.shape @sh) ++ " = " ++ show (T.toList tensor))

main :: IO ()
main = do
  scalarDerivatives
  vectorPullbacks
  gradientPaths
  maskedValues
  linearRegression

scalarDerivatives :: IO ()
scalarDerivatives = do
  putStrLn "\nScalar values: f(x) = x * x + sin x"
  let function x = x * x + sin x
      input = 0.5 :: Double
  putStrLn ("evaluate = " ++ show (AG.evaluate function input))
  putStrLn ("backprop (value, gradient) = " ++ show (AG.backprop function input))
  putStrLn ("gradient = " ++ show (AG.gradient function input))
  putStrLn ("analytic gradient = " ++ show (2 * input + cos input))

vectorPullbacks :: IO ()
vectorPullbacks = do
  putStrLn "\nVector outputs: f(x) = x * x"
  let input = T.fromList @'[3] [1, 2, 3 :: Double]
      (value, backward) = AG.pullback (\x -> x * x) input
      (total, defaultGradient) = AG.backprop (\x -> x * x) input
  showTensor "output" value
  showTensor "backprop output" total
  showTensor "default gradient: sum of output elements" defaultGradient
  showTensor "pullback: select the second output" (backward (T.fromList @'[3] [0, 1, 0]))
  showTensor "pullback: weighted outputs" (backward (T.fromList @'[3] [1, 2, 3]))

gradientPaths :: IO ()
gradientPaths = do
  putStrLn "\nGradient paths, constants, and indexing"
  let input = T.fromList @'[3] [1, 2, 3 :: Double]
      frozen = T.fromList @'[3] [10, 20, 30 :: Double]
      (_, (dx, dy)) = AG.backprop (\(x, y) -> T.sum (x * AG.stopGrad y)) (input, frozen)
  showTensor "gradient of sum (x * x)" (AG.gradient (\x -> T.sum (x * x)) input)
  showTensor "gradient of sum (x * stopGrad x)" (AG.gradient (\x -> T.sum (x * AG.stopGrad x)) input)
  showTensor "gradient of sum (x * constant frozen)" (AG.gradient (\x -> T.sum (x * AG.constant frozen)) input)
  showTensor "gradient with respect to x" dx
  showTensor "gradient with respect to stopped y" dy
  showTensor "gradient of index [1]" (AG.gradient (T.index @'[1]) input)
  showTensor "gradient of sum of slice [1:3]" (AG.gradient (T.sum . T.slice @'[ 'T.Range 1 3 1]) input)

maskedValues :: IO ()
maskedValues = do
  putStrLn "\nDifferentiable masks and element expressions"
  let input = T.fromList @'[4] [-2, 0, 1, 3 :: Double]
      relu x = T.select (x T.> 0) x 0
      (value, gradient) = AG.backprop relu input
      selectedSquare :: (AG.Context s) => AG.Tensor s '[4] Double -> AG.Tensor s '[] Double
      selectedSquare x = let element = T.index @'[3] x in T.scalar (element * element)
  showTensor "ReLU" value
  showTensor "ReLU gradient: zero at the boundary" gradient
  showTensor "gradient through index, element arithmetic, and scalar" (AG.gradient selectedSquare input)
  showTensor "max x 0 gradient: ties split equally" (AG.gradient (\x -> T.max x 0) input)

linearRegression :: IO ()
linearRegression = do
  putStrLn "\nFit y = 2 * x0 - 3 * x1 + 0.5"
  let input = T.fromList @'[4, 2] [-1, -1, -1, 1, 1, -1, 1, 1 :: Double]
      target = T.fromList @'[4, 1] [1.5, -4.5, 5.5, -0.5 :: Double]
      initial = Parameters (T.zeros @'[2, 1]) (T.zeros @'[1])
      (initialLoss, initialGradient) = AG.backprop (objective input target) initial
      step = T.run3 T.CPU $ \batch w b ->
        let (loss, Parameters dw db) = AG.backprop (objective batch target) (Parameters w b)
         in (loss, (w - 0.1 * dw, b - 0.1 * db))
      infer = T.run3 T.CPU $ \batch w b ->
        AG.evaluate (\parameters -> predict parameters (AG.constant batch)) (Parameters w b)
      batchData = T.run T.CPU input
      train :: Int -> T.TensorData '[2, 1] Double -> T.TensorData '[1] Double -> IO (T.TensorData '[2, 1] Double, T.TensorData '[1] Double)
      train iteration w b = do
        let (loss, (nextW, nextB)) = step batchData w b
        if iteration `mod` 10 == 0
          then showData ("step " ++ show iteration ++ ", mean squared error") loss
          else pure ()
        if iteration == 40
          then pure (w, b)
          else train (iteration + 1) nextW nextB
  showTensor "initial loss" initialLoss
  showTensor "initial gradient: weights" (weights initialGradient)
  showTensor "initial gradient: bias" (bias initialGradient)
  (trainedWeights, trainedBias) <- train 0 (T.run T.CPU (weights initial)) (T.run T.CPU (bias initial))
  showData "learned weights: expected [2, -3]" trainedWeights
  showData "learned bias: expected [0.5]" trainedBias
  showData "predictions through the same predict function using evaluate" (infer batchData trainedWeights trainedBias)
  showTensor "targets" target
  showTensor "gradient with respect to input features" $
    AG.gradient
      (\features -> T.sum (predict (AG.constant (Parameters (T.fromData trainedWeights) (T.fromData trainedBias))) features))
      input
