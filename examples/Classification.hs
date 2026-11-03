{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Monad (foldM, when)
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.Tensor as T
import Text.Printf (printf)

predict ::
  (T.TensorOps t Double, Num (t '[12, 3] Double)) =>
  NN.Dense 2 3 t Double ->
  t '[12, 2] Double ->
  t '[12, 3] Double
predict = NN.dense

trainingStep ::
  NN.Dense 2 3 T.Tensor Double ->
  T.Tensor '[12, 2] Double ->
  T.Tensor '[12] Int ->
  (T.Tensor '[] Double, (T.Tensor '[] Double, NN.Dense 2 3 T.Tensor Double))
trainingStep parameters features targets =
  let (loss, gradients) = AG.backprop (\p -> NN.unsafeCrossEntropy @1 T.mean (predict p $ AG.constant features) targets) parameters
      updated = NN.evalState (NN.optimizerStep (NN.SGD 0.1) parameters gradients) ()
      accuracy = NN.accuracy @1 (predict parameters features) targets
   in (loss, (accuracy, updated))

main :: IO ()
main = do
  let features = T.fromListData @'[12, 2] [-2, -1, -1, -2, -2, -2, -1, -1, 2, -1, 1, -2, 2, -2, 1, -1, 0, 2, -1, 2, 1, 2, 0, 3]
      targets = T.fromListData @'[12] [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
      initial = T.run T.CPU $ NN.Dense (T.zeros @'[2, 3] @Double) (T.zeros @'[3])
      step = T.run3 T.CPU trainingStep

  putStrLn "Three-class classification: Dense 2 -> 3 logits"

  trained <-
    foldM
      ( \parameters iteration -> do
          let (loss, (accuracy, updated)) = step parameters features targets
          when (iteration `mod` 20 == 0) $
            printf "Step %3d | loss %.4f | accuracy %.1f%%\n" iteration (sum $ T.toList loss) (100 * sum (T.toList accuracy))
          pure updated
      )
      initial
      [0 .. 100 :: Int]

  let logits = T.fromData $ T.run2 T.CPU predict trained features
      probabilities = T.softmax @1 logits
      selected = T.unsafeGather @1 probabilities $ T.unsqueeze @1 $ T.fromData targets
      oneHot = T.fromList @'[12, 3] $ concatMap (\label -> [if label == category then 1 else 0 | category <- [0 .. 2 :: Int]]) $ T.toList targets

  putStrLn $ "Predicted classes: " ++ show (T.toList $ T.run T.CPU $ T.argmax @1 logits)
  putStrLn $ "Macro F1: " ++ show (T.toList $ T.run T.CPU $ NN.f1 @1 NN.Macro logits $ T.fromData targets)
  putStrLn $ "Recall per class: " ++ show (T.toList $ T.run T.CPU $ NN.recallPerClass @1 logits $ T.fromData targets)
  putStrLn $ "Confusion matrix: " ++ show (T.toList $ T.run T.CPU $ NN.confusionMatrix @1 logits $ T.fromData targets)
  putStrLn $ "Correct-class probabilities: " ++ show (T.toList $ T.run T.CPU selected)
  putStrLn $ "Loss with probability targets: " ++ show (T.toList $ T.run T.CPU $ NN.crossEntropy @1 T.mean logits oneHot)
  putStrLn $ "Loss with integer targets: " ++ show (T.toList $ T.run T.CPU $ NN.unsafeCrossEntropy @1 T.mean logits $ T.fromData targets)

  putStrLn $ "Loss with label smoothing 0.1: " ++ show (T.toList $ T.run T.CPU $ NN.unsafeCrossEntropySmoothed @1 T.mean 0.1 logits $ T.fromData targets)

  let source = T.fromList @'[2, 3] [10, 20, 30, 40, 50, 60 :: Double]
      indices = T.fromList @'[2, 2] [2, -1, 0, 3]
  putStrLn $ "Gather with fallback -1: " ++ show (T.toList $ T.run T.CPU $ T.gatherOr @1 (-1) source indices)
