module NNTest.Layers.Recurrent.Reference (Kind (..), Parameters (..), Result (..), parameters, parameterValues, evaluate, sample, initialHidden, initialMemory, outputSeeds, objective, finiteDifferences) where

import Data.List (foldl')

-- Independent scalar implementations of the published recurrent equations.
data Kind = RNN | GRU | LSTM deriving (Eq, Show)

data Parameters = Parameters [Double] [Double] [Double] [Double]

data Result = Result {outputs :: [Double], hidden :: [Double], memory :: [Double]}

gateCount :: Kind -> Int
gateCount RNN = 1
gateCount GRU = 3
gateCount LSTM = 4

parameters :: Kind -> Parameters
parameters kind =
  let count = gateCount kind
      values n phase = [0.17 * sin (fromIntegral i * 0.73 + phase) | i <- [1 .. n]]
   in Parameters (values (4 * count) 0.2) (values (4 * count) 0.9) (values (2 * count) 1.7) (values (2 * count) 2.4)

parameterValues :: Parameters -> [Double]
parameterValues (Parameters w r b c) = w ++ r ++ b ++ c

sample :: Int -> Int -> [Double]
sample time batch = [sin (fromIntegral i * 0.2) * 0.4 | i <- [1 .. time * batch * 2]]

initialHidden :: Int -> [Double]
initialHidden batch = take (batch * 2) $ cycle [0.13, -0.21]

initialMemory :: Int -> [Double]
initialMemory batch = take (batch * 2) $ cycle [-0.17, 0.24]

outputSeeds :: Int -> Int -> [Double]
outputSeeds time batch = [cos (fromIntegral i * 0.3) | i <- [1 .. time * batch * 2]]

-- Each row vector multiplies weights stored in input/output order.
project :: Int -> [Double] -> [Double] -> [Double] -> [Double]
project width weights bias input = [sum (zipWith (*) input [weights !! (i * width + j) | i <- [0, 1]]) + bias !! j | j <- [0 .. width - 1]]

cell :: Kind -> Parameters -> [Double] -> [Double] -> [Double] -> ([Double], [Double])
cell kind (Parameters w r b c) x h oldMemory =
  let width = 2 * gateCount kind
      xp = project width w b x
      hp = project width r c h
      gate i = take 2 . drop (i * 2)
      sigmoid value = 1 / (1 + exp (-value))
      both = zipWith (+) xp hp
   in case kind of
        RNN -> (map tanh both, [])
        GRU ->
          let reset = map sigmoid $ gate 0 both
              update = map sigmoid $ gate 1 both
              candidate = map tanh $ zipWith (+) (gate 2 xp) $ zipWith (*) reset $ gate 2 hp
           in (zipWith3 (\z n previous -> (1 - z) * n + z * previous) update candidate h, [])
        LSTM ->
          let i = map sigmoid $ gate 0 both
              f = map sigmoid $ gate 1 both
              g = map tanh $ gate 2 both
              o = map sigmoid $ gate 3 both
              nextMemory = zipWith (+) (zipWith (*) f oldMemory) (zipWith (*) i g)
           in (zipWith (*) o $ map tanh nextMemory, nextMemory)

evaluate :: Kind -> Parameters -> Int -> Int -> [Double] -> [Double] -> [Double] -> Result
evaluate kind p time batch inputs h0 c0 =
  let step (Result previous h c) i =
        let rows = [cell kind p (take 2 $ drop (i * batch * 2 + row * 2) inputs) (take 2 $ drop (row * 2) h) (take 2 $ drop (row * 2) c) | row <- [0 .. batch - 1]]
            nextH = concatMap fst rows
            nextC = concatMap snd rows
         in Result (previous ++ nextH) nextH nextC
   in foldl' step (Result [] h0 c0) [0 .. time - 1]

-- A weighted objective includes every output and both final state paths.
objective :: Kind -> [Double] -> Double
objective kind vector =
  let count = gateCount kind
      (w, rest1) = splitAt (4 * count) vector
      (r, rest2) = splitAt (4 * count) rest1
      (b, rest3) = splitAt (2 * count) rest2
      (c, rest4) = splitAt (2 * count) rest3
      (x, rest5) = splitAt 4 rest4
      (h, initialC) = splitAt 2 rest5
      result = evaluate kind (Parameters w r b c) 2 1 x h initialC
   in sum (zipWith (*) (outputSeeds 2 1) (outputs result)) + 0.3 * sum (hidden result) - 0.2 * sum (memory result)

finiteDifferences :: Kind -> [Double] -> [Double]
finiteDifferences kind vector =
  let epsilon = 1e-5
      perturb index change = zipWith (\i x -> if i == index then x + change else x) [0 :: Int ..] vector
   in [(objective kind (perturb i epsilon) - objective kind (perturb i $ negate epsilon)) / (2 * epsilon) | i <- [0 .. length vector - 1]]
