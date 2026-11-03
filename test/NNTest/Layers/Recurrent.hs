{-# LANGUAGE Arrows #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module NNTest.Layers.Recurrent (tests) where

import Control.Arrow (arr)
import Control.Monad.Trans.State.Strict (runState)
import qualified Hedgehog as H
import qualified NNTest.Layers.Recurrent.Reference as Ref
import qualified Synapse.Autograd as AG
import qualified Synapse.NN as NN
import qualified Synapse.NN.Weights as Weights
import qualified Synapse.Tensor as T
import qualified TensorTest.Compiler as Compiler
import TensorTest.Support (approxList)
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.Hedgehog (testProperty)

rnnParameters :: NN.RNN 2 2 T.Tensor Double
rnnParameters = let Ref.Parameters w r b c = Ref.parameters Ref.RNN in NN.RNN (T.fromList w) (T.fromList r) (T.fromList b) (T.fromList c)

type RNNInput t = (NN.RNN 2 2 t Double, (t '[2, 1, 2] Double, t '[1, 2] Double))

type RNNOutput t = (t '[2, 1, 2] Double, t '[1, 2] Double)

rnnInput :: RNNInput T.Tensor
rnnInput = (rnnParameters, (T.fromList $ Ref.sample 2 1, T.fromList $ Ref.initialHidden 1))

rnnForward :: (T.TensorOps t Double) => RNNInput t -> RNNOutput t
rnnForward (p, (xs, initial)) = runState (NN.rnn p xs) initial

rnnGradientValues :: T.WithTensor T.TensorData (RNNInput T.Tensor) -> [Double]
rnnGradientValues (p, (xs, initial)) = concat [T.toList $ NN.rnnInputWeights p, T.toList $ NN.rnnHiddenWeights p, T.toList $ NN.rnnInputBias p, T.toList $ NN.rnnHiddenBias p, T.toList xs, T.toList initial]

checkRNNGradient :: H.PropertyT IO ()
checkRNNGradient = do
  let seed :: RNNOutput T.Tensor
      seed = (T.fromList $ Ref.outputSeeds 2 1, 0.3)
      (output, backward) = AG.pullback rnnForward rnnInput
      (actual, gradient) = T.run T.CPU (output, backward seed)
      expected = Ref.evaluate Ref.RNN (Ref.parameters Ref.RNN) 2 1 (Ref.sample 2 1) (Ref.initialHidden 1) []
      vector = Ref.parameterValues (Ref.parameters Ref.RNN) ++ Ref.sample 2 1 ++ Ref.initialHidden 1
  approxList (T.toList $ fst actual) $ Ref.outputs expected
  approxList (T.toList $ snd actual) $ Ref.hidden expected
  approxList (rnnGradientValues gradient) $ Ref.finiteDifferences Ref.RNN vector

gruParameters :: NN.GRU 2 2 T.Tensor Double
gruParameters = let Ref.Parameters w r b c = Ref.parameters Ref.GRU in NN.GRU (T.fromList w) (T.fromList r) (T.fromList b) (T.fromList c)

type GRUInput t = (NN.GRU 2 2 t Double, (t '[2, 1, 2] Double, t '[1, 2] Double))

type GRUOutput t = (t '[2, 1, 2] Double, t '[1, 2] Double)

gruInput :: GRUInput T.Tensor
gruInput = (gruParameters, (T.fromList $ Ref.sample 2 1, T.fromList $ Ref.initialHidden 1))

gruForward :: (T.TensorOps t Double) => GRUInput t -> GRUOutput t
gruForward (p, (xs, initial)) = runState (NN.gru p xs) initial

gruGradientValues :: T.WithTensor T.TensorData (GRUInput T.Tensor) -> [Double]
gruGradientValues (p, (xs, initial)) = concat [T.toList $ NN.gruInputWeights p, T.toList $ NN.gruHiddenWeights p, T.toList $ NN.gruInputBias p, T.toList $ NN.gruHiddenBias p, T.toList xs, T.toList initial]

checkGRUGradient :: H.PropertyT IO ()
checkGRUGradient = do
  let seed :: GRUOutput T.Tensor
      seed = (T.fromList $ Ref.outputSeeds 2 1, 0.3)
      (output, backward) = AG.pullback gruForward gruInput
      (actual, gradient) = T.run T.CPU (output, backward seed)
      expected = Ref.evaluate Ref.GRU (Ref.parameters Ref.GRU) 2 1 (Ref.sample 2 1) (Ref.initialHidden 1) []
      vector = Ref.parameterValues (Ref.parameters Ref.GRU) ++ Ref.sample 2 1 ++ Ref.initialHidden 1
  approxList (T.toList $ fst actual) $ Ref.outputs expected
  approxList (T.toList $ snd actual) $ Ref.hidden expected
  approxList (gruGradientValues gradient) $ Ref.finiteDifferences Ref.GRU vector

lstmParameters :: NN.LSTM 2 2 T.Tensor Double
lstmParameters = let Ref.Parameters w r b c = Ref.parameters Ref.LSTM in NN.LSTM (T.fromList w) (T.fromList r) (T.fromList b) (T.fromList c)

type LSTMInput t = (NN.LSTM 2 2 t Double, (t '[2, 1, 2] Double, NN.LSTMState 1 2 t Double))

type LSTMOutput t = (t '[2, 1, 2] Double, NN.LSTMState 1 2 t Double)

lstmInput :: LSTMInput T.Tensor
lstmInput = (lstmParameters, (T.fromList $ Ref.sample 2 1, NN.LSTMState (T.fromList $ Ref.initialHidden 1) (T.fromList $ Ref.initialMemory 1)))

lstmForward :: (T.TensorOps t Double) => LSTMInput t -> LSTMOutput t
lstmForward (p, (xs, initial)) = runState (NN.lstm p xs) initial

lstmGradientValues :: T.WithTensor T.TensorData (LSTMInput T.Tensor) -> [Double]
lstmGradientValues (p, (xs, initial)) = concat [T.toList $ NN.lstmInputWeights p, T.toList $ NN.lstmHiddenWeights p, T.toList $ NN.lstmInputBias p, T.toList $ NN.lstmHiddenBias p, T.toList xs, T.toList $ NN.lstmHiddenState initial, T.toList $ NN.lstmCellState initial]

checkLSTMGradient :: H.PropertyT IO ()
checkLSTMGradient = do
  let seed :: LSTMOutput T.Tensor
      seed = (T.fromList $ Ref.outputSeeds 2 1, NN.LSTMState 0.3 (-0.2))
      (output, backward) = AG.pullback lstmForward lstmInput
      (actual, gradient) = T.run T.CPU (output, backward seed)
      expected = Ref.evaluate Ref.LSTM (Ref.parameters Ref.LSTM) 2 1 (Ref.sample 2 1) (Ref.initialHidden 1) (Ref.initialMemory 1)
      vector = Ref.parameterValues (Ref.parameters Ref.LSTM) ++ Ref.sample 2 1 ++ Ref.initialHidden 1 ++ Ref.initialMemory 1
  approxList (T.toList $ fst actual) $ Ref.outputs expected
  approxList (T.toList $ NN.lstmHiddenState $ snd actual) $ Ref.hidden expected
  approxList (T.toList $ NN.lstmCellState $ snd actual) $ Ref.memory expected
  approxList (lstmGradientValues gradient) $ Ref.finiteDifferences Ref.LSTM vector

tests :: TestTree
tests =
  testGroup
    "recurrent layers"
    [ testProperty "RNN cell agrees with scalar gate equations" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] $ Ref.sample 1 2
            initial = T.fromList @'[2, 2] $ Ref.initialHidden 2
            (output, final) = T.run T.CPU $ runState (NN.rnnCell rnnParameters input) initial
            expected = Ref.evaluate Ref.RNN (Ref.parameters Ref.RNN) 1 2 (Ref.sample 1 2) (Ref.initialHidden 2) []
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList final) $ Ref.hidden expected,
      testProperty "RNN sequence agrees with batched scalar recurrence" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3, 2, 2] $ Ref.sample 3 2
            initial = T.fromList @'[2, 2] $ Ref.initialHidden 2
            (output, final) = T.run T.CPU $ runState (NN.rnn rnnParameters input) initial
            expected = Ref.evaluate Ref.RNN (Ref.parameters Ref.RNN) 3 2 (Ref.sample 3 2) (Ref.initialHidden 2) []
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList final) $ Ref.hidden expected,
      testProperty "RNN gradients cover every weight, bias, input and initial state" $ H.withTests 1 $ H.property checkRNNGradient,
      testProperty "GRU cell agrees with scalar gate equations" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] $ Ref.sample 1 2
            initial = T.fromList @'[2, 2] $ Ref.initialHidden 2
            (output, final) = T.run T.CPU $ runState (NN.gruCell gruParameters input) initial
            expected = Ref.evaluate Ref.GRU (Ref.parameters Ref.GRU) 1 2 (Ref.sample 1 2) (Ref.initialHidden 2) []
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList final) $ Ref.hidden expected,
      testProperty "GRU sequence agrees with batched scalar recurrence" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3, 2, 2] $ Ref.sample 3 2
            initial = T.fromList @'[2, 2] $ Ref.initialHidden 2
            (output, final) = T.run T.CPU $ runState (NN.gru gruParameters input) initial
            expected = Ref.evaluate Ref.GRU (Ref.parameters Ref.GRU) 3 2 (Ref.sample 3 2) (Ref.initialHidden 2) []
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList final) $ Ref.hidden expected,
      testProperty "GRU gradients cover every weight, bias, input and initial state" $ H.withTests 1 $ H.property checkGRUGradient,
      testProperty "LSTM cell agrees with scalar gate equations" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[2, 2] $ Ref.sample 1 2
            initial = NN.LSTMState (T.fromList $ Ref.initialHidden 2) (T.fromList $ Ref.initialMemory 2)
            (output, final) = T.run T.CPU $ runState (NN.lstmCell lstmParameters input) initial
            expected = Ref.evaluate Ref.LSTM (Ref.parameters Ref.LSTM) 1 2 (Ref.sample 1 2) (Ref.initialHidden 2) (Ref.initialMemory 2)
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList $ NN.lstmHiddenState final) $ Ref.hidden expected
        approxList (T.toList $ NN.lstmCellState final) $ Ref.memory expected,
      testProperty "LSTM sequence agrees with batched scalar recurrence" $ H.withTests 1 $ H.property $ do
        let input = T.fromList @'[3, 2, 2] $ Ref.sample 3 2
            initial = NN.LSTMState (T.fromList $ Ref.initialHidden 2) (T.fromList $ Ref.initialMemory 2)
            (output, final) = T.run T.CPU $ runState (NN.lstm lstmParameters input) initial
            expected = Ref.evaluate Ref.LSTM (Ref.parameters Ref.LSTM) 3 2 (Ref.sample 3 2) (Ref.initialHidden 2) (Ref.initialMemory 2)
        approxList (T.toList output) $ Ref.outputs expected
        approxList (T.toList $ NN.lstmHiddenState final) $ Ref.hidden expected
        approxList (T.toList $ NN.lstmCellState final) $ Ref.memory expected,
      testProperty "LSTM gradients cover every weight, bias, input and initial state" $ H.withTests 1 $ H.property checkLSTMGradient,
      testProperty "empty LSTM sequences preserve both states and their gradients" $ H.withTests 1 $ H.property $ do
        let initial = NN.LSTMState (T.fromList @'[1, 2] [0.2, -0.3]) (T.fromList @'[1, 2] [0.4, -0.5])
            (result, (dp, (dx, ds))) = AG.backprop (\(p, (xs, s)) -> runState (NN.lstm p xs) s) (lstmParameters, (T.zeros @'[0, 1, 2], initial))
            ((output, final), (grad, (inputGradient, stateGradient))) = T.run T.CPU (result, (dp, (dx, ds)))
        T.toList output H.=== []
        T.toList (NN.lstmHiddenState final) H.=== [0.2, -0.3]
        T.toList (NN.lstmCellState final) H.=== [0.4, -0.5]
        concat [T.toList $ NN.lstmInputWeights grad, T.toList $ NN.lstmHiddenWeights grad, T.toList $ NN.lstmInputBias grad, T.toList $ NN.lstmHiddenBias grad] H.=== replicate 48 0
        T.toList inputGradient H.=== []
        T.toList (NN.lstmHiddenState stateGradient) H.=== [1, 1]
        T.toList (NN.lstmCellState stateGradient) H.=== [1, 1],
      testProperty "empty RNN and GRU sequences preserve hidden state" $ H.withTests 1 $ H.property $ do
        let initial = T.fromList @'[1, 2] [0.2, -0.3 :: Double]
            (ro, rh) = T.run T.CPU $ runState (NN.rnn rnnParameters $ T.zeros @'[0, 1, 2]) initial
            (go, gh) = T.run T.CPU $ runState (NN.gru gruParameters $ T.zeros @'[0, 1, 2]) initial
        T.toList ro H.=== []
        T.toList go H.=== []
        T.toList rh H.=== [0.2, -0.3]
        T.toList gh H.=== [0.2, -0.3],
      testProperty "state carries gradients across consecutive LSTM chunks" $ H.withTests 1 $ H.property $ do
        let chunks (p, (xs, s)) =
              runState
                ( do
                    first <- NN.lstm p $ T.slice @'[ 'T.Range 0 1 1, 'T.All, 'T.All] xs
                    second <- NN.lstm p $ T.slice @'[ 'T.Range 1 2 1, 'T.All, 'T.All] xs
                    pure $ T.concat @0 first second
                )
                s
            seed = (T.fromList $ Ref.outputSeeds 2 1, NN.LSTMState 0.3 (-0.2)) :: LSTMOutput T.Tensor
            (full, backward) = AG.pullback lstmForward lstmInput
            (chunked, chunkBackward) = AG.pullback chunks lstmInput
            ((ys, final), gradient) = T.run T.CPU (full, backward seed)
            ((cy, cs), cg) = T.run T.CPU (chunked, chunkBackward seed)
        approxList (T.toList cy) $ T.toList ys
        approxList (T.toList $ NN.lstmHiddenState cs) $ T.toList $ NN.lstmHiddenState final
        approxList (T.toList $ NN.lstmCellState cs) $ T.toList $ NN.lstmCellState final
        approxList (lstmGradientValues cg) $ lstmGradientValues gradient,
      testProperty "prepared LSTM accepts changing parameters, inputs and initial states" $ H.withTests 1 $ H.property $ do
        let execute = T.run1 T.CPU $ \arguments -> AG.backprop lstmForward arguments
            material = T.run T.CPU lstmInput
            first@(ys, _) = fst $ execute material
            (p, (xs, s)) = lstmInput
            changed = (p {NN.lstmInputBias = NN.lstmInputBias p + 0.1}, (xs + 0.05, NN.LSTMState (NN.lstmHiddenState s + 0.02) (NN.lstmCellState s - 0.03)))
            second@(newOutputs, _) = fst $ execute $ T.run T.CPU changed
            (expected, expectedState) = T.run T.CPU $ lstmForward changed
        H.assert $ T.toList ys /= T.toList newOutputs
        approxList (T.toList $ fst first) $ Ref.outputs $ Ref.evaluate Ref.LSTM (Ref.parameters Ref.LSTM) 2 1 (Ref.sample 2 1) (Ref.initialHidden 1) (Ref.initialMemory 1)
        approxList (T.toList $ fst second) $ T.toList expected
        approxList (T.toList $ NN.lstmCellState $ snd second) $ T.toList $ NN.lstmCellState expectedState,
      testProperty "recurrent layers accept empty batches and zero hidden width" $ H.withTests 1 $ H.property $ do
        let (r, _) = T.run T.CPU $ runState (NN.rnn rnnParameters $ T.zeros @'[2, 0, 2]) (T.zeros @'[0, 2])
            (g, _) = T.run T.CPU $ runState (NN.gru gruParameters $ T.zeros @'[2, 0, 2]) (T.zeros @'[0, 2])
            (l, _) = T.run T.CPU $ runState (NN.lstm lstmParameters $ T.zeros @'[2, 0, 2]) (NN.LSTMState (T.zeros @'[0, 2]) (T.zeros @'[0, 2]))
            empty = NN.LSTM (T.zeros @'[2, 0]) (T.zeros @'[0, 0]) (T.zeros @'[0]) (T.zeros @'[0]) :: NN.LSTM 2 0 T.Tensor Double
            (z, _) = T.run T.CPU $ runState (NN.lstm empty $ T.ones @'[2, 1, 2]) (NN.LSTMState (T.zeros @'[1, 0]) (T.zeros @'[1, 0]))
        map T.toList [r, g, l] H.=== [[], [], []]
        T.toList z H.=== [],
      testProperty "LSTM supports Float parameters and saturated gates without NaN gradients" $ H.withTests 1 $ H.property $ do
        let p = NN.LSTM (T.zeros @'[1, 4]) (T.zeros @'[1, 4]) (T.fromList @'[4] [-1000, 1000, 0, -1000]) (T.zeros @'[4]) :: NN.LSTM 1 1 T.Tensor Float
            input = (p, (T.ones @'[1, 1, 1], NN.LSTMState (T.zeros @'[1, 1]) (T.ones @'[1, 1])))
            ((output, final), (gradient, _)) = T.run T.CPU $ AG.backprop (\(layer, (xs, s)) -> runState (NN.lstm layer xs) s) input
        T.toList output H.=== [0]
        T.toList (NN.lstmCellState final) H.=== [1]
        H.assert $ all (all (\x -> not $ isNaN x || isInfinite x)) [T.toList $ NN.lstmInputWeights gradient, T.toList $ NN.lstmHiddenWeights gradient, T.toList $ NN.lstmInputBias gradient, T.toList $ NN.lstmHiddenBias gradient],
      testProperty "recurrent parameters materialize and round-trip through weight dictionaries" $ H.withTests 1 $ H.property $ do
        let p = T.run T.CPU (rnnParameters, gruParameters, lstmParameters)
            dictionary = NN.extractWeights p
        length (Weights.weightsToList dictionary) H.=== 12
        (_, _, restored) <- H.evalEither $ NN.loadWeights dictionary p
        T.toList (NN.lstmInputWeights restored) H.=== T.toList (NN.lstmInputWeights $ let (_, _, l) = p in l),
      testProperty "LSTM state actions compose through Forward and standard arrows" $ H.withTests 1 $ H.property $ do
        let network :: NN.Forward (NN.LSTMState 1 2 T.Tensor Double) (T.Tensor '[1, 2] Double) (T.Tensor '[1, 2] Double)
            network = proc x -> do
              y <- NN.Forward (NN.lstmCell lstmParameters) -< x
              arr tanh -< y
            (p, (xs, s)) = lstmInput
            input = T.slice @'[ 'T.At 0, 'T.All, 'T.All] xs
            (actual, as) = T.run T.CPU $ runState (NN.forward network input) s
            (expected, es) = T.run T.CPU $ runState (NN.lstmCell p input) s
        approxList (T.toList actual) $ map tanh $ T.toList expected
        approxList (T.toList $ NN.lstmCellState as) $ T.toList $ NN.lstmCellState es,
      testProperty "SGD updates recurrent records and reduces a sequence objective" $ H.withTests 1 $ H.property $ do
        let (_, (xs, s)) = rnnInput
            execute = T.run1 T.CPU $ \p ->
              let (loss, gradient) = AG.backprop (\parameters -> T.sum $ NN.evalState (NN.rnn parameters $ AG.constant xs) (AG.constant s) ** 2) p
                  updated = NN.evalState (NN.optimizerStep (NN.SGD 0.01) p gradient) ()
               in (loss, updated)
            (before, next) = execute $ T.run T.CPU rnnParameters
            (after, _) = execute next
        case (T.toList before, T.toList after) of
          ([initialLoss], [nextLoss]) -> H.assert $ nextLoss < initialLoss
          _ -> H.failure,
      testProperty "LSTM rejects an incorrect input feature count" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/LSTMFeatures.hs" "Couldn't match type",
      testProperty "GRU rejects an incorrect recurrent state width" $ H.withTests 1 $ Compiler.rejects "test/NNTest/Fixtures/Rejected/GRUState.hs" "Couldn't match type"
    ]
