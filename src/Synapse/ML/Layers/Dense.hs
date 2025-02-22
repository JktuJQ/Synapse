{- | Provides dense layer implementation.

@Dense@ datatype represents densely-connected neural network layer and
it performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}


module Synapse.ML.Layers.Dense
    ( -- * @Dense@ datatype

      Dense (Dense, denseWeights, denseBias)
    , denseLayer
    ) where


import Synapse.ML.Layers.Layer (AbstractLayer(..), LayerConfiguration)

import Synapse.Autograd (Symbol, symbol, matMul)

import Synapse.LinearAlgebra (Indexable (unsafeIndex))

import Synapse.LinearAlgebra.Vec (Vec)
import qualified Synapse.LinearAlgebra.Vec as V

import Synapse.LinearAlgebra.Mat (Mat)
import qualified Synapse.LinearAlgebra.Mat as M


{- | @Dense@ datatype represents densely-connected neural network layer.

@Dense@ performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}
data Dense a = Dense
    { denseWeights :: Mat a          -- ^ Matrix that represents weights of dense layer.
    , denseBias    :: Maybe (Vec a)  -- ^ Vector that represents bias of dense layer.
    }

instance Functor Dense where
    fmap f (Dense weights bias) = Dense (fmap f weights) (fmap (fmap f) bias)

-- | Creates symbol for weights.
weightsSymbol :: String -> Mat a -> Symbol (Mat a)
weightsSymbol prefix = symbol (prefix ++ "1")

-- | Creates matrix that corresponds to bias (bias rows stacked on each other).
biasToMat :: Int -> Vec a -> Mat a
biasToMat rows bias = M.generate (rows, V.size bias) $ \(_, c) -> unsafeIndex bias c

-- | Creates symbol for bias.
biasSymbol :: String -> Int -> Vec a -> Symbol (Mat a)
biasSymbol prefix rows = symbol (prefix ++ "2") . biasToMat rows

instance AbstractLayer Dense where
    inputSize = Just . M.nRows . denseWeights
    outputSize = Just . M.nCols . denseWeights

    getParameters (Dense weights Nothing) = [weights]
    getParameters (Dense weights (Just bias)) = [weights, biasToMat (M.nRows weights) bias]

    updateParameters (Dense _ Nothing) [weights'] = Dense weights' Nothing
    updateParameters (Dense _ (Just _)) [weights', biasMat'] = Dense weights' (Just $ M.indexRow biasMat' 0)
    updateParameters _ _ = error "Parameters update failed - wrong amount of parameters was given"

    symbolicForward prefix (Dense weights Nothing) input = input `matMul` weightsSymbol prefix weights
    symbolicForward prefix (Dense weights (Just bias)) input = input `matMul` weightsSymbol prefix weights + biasSymbol prefix (M.nRows weights) bias

    forward (Dense weights Nothing) input = input `M.matMul` weights
    forward (Dense weights (Just bias)) input = input `M.matMul` weights + biasToMat (M.nRows weights) bias


-- | Creates configuration for dense layer.
denseLayer :: Num a => Int -> LayerConfiguration (Dense a)
denseLayer neurons input = Dense (M.replicate (input, neurons) 0) (Just $ V.replicate neurons 0)
