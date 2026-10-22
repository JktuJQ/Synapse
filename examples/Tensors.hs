{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import qualified Synapse.Tensor as T

showTensor :: (T.Shape sh, T.DType a, Show a) => String -> T.Tensor sh a -> IO ()
showTensor label = showData label . T.run T.CPU

showData :: forall sh a. (T.Shape sh, T.DType a, Show a) => String -> T.TensorData sh a -> IO ()
showData label tensor = do
  putStrLn (label ++ " " ++ show (T.shape @sh))
  print (T.toList tensor)

showElement :: (T.DType a, Show a) => String -> T.TensorElement a -> IO ()
showElement label = showTensor label . T.scalar

main :: IO ()
main = do
  construction
  elementwise
  shapes
  indexing
  linearAlgebra
  reductions
  controlFlow
  preparedFunctions
  classification

construction :: IO ()
construction = do
  putStrLn "\nConstruction"
  let matrix = T.fromList @'[2, 3] [1 .. 6 :: Double]
      generated = T.fromGen @'[2, 3] (\ix -> T.coordinate @0 ix * 10 + T.coordinate @1 ix)
      materialized = T.run T.CPU matrix
  showTensor "fromList" matrix
  showTensor "fromGen: row * 10 + column" generated
  showTensor "zeros" (T.zeros @'[2, 3] @Double)
  showTensor "ones" (T.ones @'[3] @Double)
  showTensor "full" (T.full @'[2] True)
  showTensor "arange" (T.arange @5 @Int 0 2)
  showTensor "linspace" (T.linspace @5 @Double 0 1)
  showTensor "eye" (T.eye @2 @3 @Double)
  showTensor "fromData: reuse a CPU result" (T.fromData materialized + 1)

elementwise :: IO ()
elementwise = do
  putStrLn "\nElementwise operations"
  let x = T.fromList @'[2, 3] [-2, -1, 0, 1, 2, 3 :: Double]
      mask = (x T.> 0) T.&& (x T.< 3)
      integers = T.fromList @'[4] [-7, -2, 2, 7 :: Int]
      (unzippedValues, unzippedMask) = T.unzip (T.zip x mask)
      (_, _, unzippedThird) = T.unzip3 (T.zip3 x mask (x + 1))
  showTensor "x * 2 + 1" (x * 2 + 1)
  showTensor "map: square" (T.map (\value -> value * value) x)
  showTensor "map: Int to Double" (T.map T.fromIntegral integers :: T.Tensor '[4] Double)
  showTensor "zip" (T.zip x mask)
  showTensor "zip3" (T.zip3 x mask (x + 1))
  showTensor "unzip: values" unzippedValues
  showTensor "unzip: mask" unzippedMask
  showTensor "unzip3: third component" unzippedThird
  showTensor "zipWith: squared difference" (T.zipWith (\a b -> (a - b) * (a - b)) x (x + 1))
  showTensor "zipWith3: masked sum" (T.zipWith3 (\a b flag -> T.select flag (a + b) 0) x (x + 1) mask)
  showTensor "0 < x < 3" mask
  showTensor "select: keep matching values" (T.select mask x 0)
  showTensor "quot: truncate toward zero" (T.quot integers 3)
  showTensor "rem: remainder" (T.rem integers 3)

shapes :: IO ()
shapes = do
  putStrLn "\nShapes and axes"
  let matrix = T.fromList @'[2, 3] [1 .. 6 :: Double]
      row = T.fromList @'[3] [10, 20, 30]
      cube = T.reshape @'[1, 2, 3] matrix
  showTensor "broadcast row and add" (matrix + T.broadcast row)
  showTensor "reshape" (T.reshape @'[3, 2] matrix)
  showTensor "unsqueeze" (T.unsqueeze @0 matrix)
  showTensor "squeeze" (T.squeeze @0 cube)
  showTensor "transpose" (T.transpose matrix)
  showTensor "permuteAxes" (T.permuteAxes @'[2, 0, 1] cube)
  showTensor "concat along rows" (T.concat @0 matrix matrix)
  showTensor "stack along a new leading axis" (T.stack @0 matrix matrix)

indexing :: IO ()
indexing = do
  putStrLn "\nIndexing and slices"
  let matrix = T.fromList @'[2, 3] [1 .. 6 :: Int]
  showElement "index [1, 2]" (T.index @'[1, 2] matrix)
  showTensor "slice: second row" (T.slice @'[ 'T.At 1, 'T.All] matrix)
  showTensor "slice: every other column" (T.slice @'[ 'T.All, 'T.Range 0 3 2] matrix)
  showTensor "slice: reverse columns" (T.slice @'[ 'T.All, 'T.RangeReverse 0 3 1] matrix)
  showTensor "fromGen: read using a complete index" (T.fromGen @'[2, 3] (matrix T.!!))

linearAlgebra :: IO ()
linearAlgebra = do
  putStrLn "\nLinear algebra"
  let vector = T.fromList @'[3] [1, 2, 3 :: Double]
      lhs = T.fromList @'[2, 3] [1 .. 6 :: Double]
      rhs = T.fromList @'[3, 2] [1 .. 6]
  showTensor "dot" (T.dot vector vector)
  showTensor "matmul" (T.matmul lhs rhs)
  showTensor "batched matmul" (T.matmul (T.stack @0 lhs lhs) (T.stack @0 rhs rhs))

reductions :: IO ()
reductions = do
  putStrLn "\nReductions and normalization"
  let matrix = T.fromList @'[2, 3] [1000, 1001, 1002, -1002, -1001, -1000 :: Double]
      mask = matrix T.> 0
  showTensor "sum" (T.sum matrix)
  showTensor "mean" (T.mean matrix)
  showTensor "minimum" (T.minimum matrix)
  showTensor "maximum" (T.maximum matrix)
  showTensor "sumAxes: row sums" (T.sumAxes @'[1] matrix)
  showTensor "meanAxes: column means" (T.meanAxes @'[0] matrix)
  showTensor "argmax: position within each row" (T.argmax @1 matrix)
  showTensor "argmin: position within each row" (T.argmin @1 matrix)
  showTensor "all positive" (T.all mask)
  showTensor "any positive per row" (T.anyAxes @'[1] mask)
  showTensor "softmax per row" (T.softmax @1 matrix)
  showTensor "logSoftmax per row" (T.logSoftmax @1 matrix)
  showTensor "logSumExp" (T.logSumExp matrix)
  showTensor "logSumExpAxes per row" (T.logSumExpAxes @'[1] matrix)

controlFlow :: IO ()
controlFlow = do
  putStrLn "\nElement expressions and control flow"
  let integers = T.fromList @'[4] [-2, 0, 3, 5 :: Int]
      factorial :: T.TensorElement Int -> T.TensorElement Int
      factorial limit =
        T.snd $
          T.while
            (\state -> T.fst state T.<= limit)
            (\state -> T.pair (T.fst state + 1) (T.snd state * T.fst state))
            (T.pair 1 1)
  showTensor "select within map: absolute value" (T.map (\x -> T.select (x T.< 0) (negate x) x) integers)
  showTensor "cond: choose a whole tensor" (T.cond (T.any (integers T.< 0)) (negate integers) integers)
  showTensor "while with a pair state: factorials" (T.fromGen @'[6] (factorial . T.coordinate @0))
  showTensor "iterate: powers of two" (T.fromGen @'[5] (\ix -> T.iterate (T.coordinate @0 ix) (* 2) (1 :: T.TensorElement Int)))

preparedFunctions :: IO ()
preparedFunctions = do
  putStrLn "\nPrepared tensor functions"
  let affine = T.run1 T.CPU (\x -> x * 2 + 1)
      first = T.run T.CPU (T.fromList @'[3] [1, 2, 3 :: Double])
      second = T.run T.CPU (T.fromList @'[3] [4, 5, 6])
      multiply = T.run2 T.CPU T.matmul
      lhs = T.run T.CPU (T.fromList @'[2, 3] [1 .. 6 :: Double])
      rhs = T.run T.CPU (T.fromList @'[3, 2] [1 .. 6])
      other = T.run T.CPU (T.ones @'[3, 2])
      linear = T.run3 T.CPU (\x w b -> T.matmul x w + T.broadcast b)
      summarize = T.run1 T.CPU (\x -> (x * 2 + 1, (T.mean x, x T.> 2)))
      maskedLinear = T.run4 T.CPU (\x w b mask -> T.select mask (T.matmul x w + T.broadcast b) 0)
  putStrLn "run1: reuse with two inputs"
  print (T.toList (affine first))
  print (T.toList (affine second))
  putStrLn "run2: reuse with changing matrices"
  print (T.toList (multiply lhs rhs))
  print (T.toList (multiply lhs other))
  putStrLn "run3: inputs, weights, and bias"
  print (T.toList (linear lhs rhs (T.run T.CPU (T.fromList @'[2] [1, 2]))))
  print (T.toList (linear lhs other (T.run T.CPU (T.zeros @'[2]))))
  putStrLn "run1: nested results with different shapes and element types"
  let showSummary input = do
        let (values, (average, mask)) = summarize input
        showData "transformed values" values
        showData "input mean" average
        showData "input > 2" mask
  showSummary first
  showSummary second
  putStrLn "run: unzip into several materialized tensors"
  let (values, flags) = T.run T.CPU (T.unzip (T.zip (T.fromData first) (T.fromData first T.> 2)))
  showData "values" values
  showData "flags" flags
  putStrLn "run4: partially apply inputs, weights, and bias; vary the mask"
  let applyMask = maskedLinear lhs rhs (T.run T.CPU (T.fromList @'[2] [1, 2]))
  showData "diagonal mask" (applyMask (T.run T.CPU (T.fromList @'[2, 2] [True, False, False, True])))
  showData "all values" (applyMask (T.run T.CPU (T.full @'[2, 2] True)))

classification :: IO ()
classification = do
  putStrLn "\nLinear classifier"
  let inputs = T.fromList @'[3, 2] [2, 0, 0, 2, 1, 1]
      weights = T.fromList @'[2, 3] [1, -1, 0, -1, 1, 0]
      bias = T.fromList @'[3] [0, 0, 0.5 :: Double]
      evaluate = T.run3 T.CPU $ \x w b ->
        let scores = T.matmul x w + T.broadcast b
         in (scores, T.softmax @1 scores, T.argmax @1 scores)
      (logits, probabilities, predictions) = evaluate (T.run T.CPU inputs) (T.run T.CPU weights) (T.run T.CPU bias)
  showTensor "inputs" inputs
  showTensor "weights" weights
  showTensor "bias" bias
  showData "logits = matmul inputs weights + broadcast bias" logits
  showData "probabilities" probabilities
  showData "predicted classes" predictions
