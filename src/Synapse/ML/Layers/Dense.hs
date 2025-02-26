{- | Provides dense layer implementation.

@Dense@ datatype represents densely-connected neural network layer and
it performs following operation: @x `matMul` w + b@, where @w@ is weights and @b@ is bias (if present) of a layer.
-}


module Synapse.ML.Layers.Dense
    ( -- * @Dense@ datatype

      Dense (Dense, denseWeights, denseBias, denseConstraints)
    , layerDenseWith
    , layerDense
    ) where


import Synapse.ML.Layers.Layer (AbstractLayer(..), LayerConfiguration)
import Synapse.ML.Layers.Initializers (Initializer(Initializer), zeroes)
import Synapse.ML.Layers.Constraints (Constraint, applyConstraints)

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
    { denseWeights     :: Mat a                             -- ^ Matrix that represents weights of dense layer.
    , denseBias        :: Vec a                             -- ^ Vector that represents bias of dense layer.
    
    , denseConstraints :: ([Constraint a], [Constraint a])  -- ^ Constraints on weights and bias of dense layer.
    }

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

    getParameters (Dense weights bias _) = [weights, biasToMat (M.nRows weights) bias]

    updateParameters (Dense _ _ constraints) [weights', biasMat'] =
        Dense (applyConstraints (fst constraints) weights') (M.indexRow (applyConstraints (snd constraints) biasMat') 0) constraints
    updateParameters _ _ = error "Parameters update failed - wrong amount of parameters was given"

    symbolicForward prefix (Dense weights bias _) input =
        input `matMul` weightsSymbol prefix weights + biasSymbol prefix (M.nRows weights) bias

-- | Creates configuration of dense layer.
layerDenseWith
    :: Num a
    => (Initializer a, [Constraint a])  -- ^ Weights initializer and constraints.
    -> (Initializer a, [Constraint a])  -- ^ Bias initializer and constraints.
    -> Int                              -- ^ Amount of neurons.
    -> LayerConfiguration (Dense a)
layerDenseWith (Initializer weightsInitializer, weightsConstraints) (Initializer biasInitializer, biasConstraints) neurons input =
    Dense (weightsInitializer (input, neurons)) (M.indexRow (biasInitializer (1, neurons)) 0) (weightsConstraints, biasConstraints)

-- | Creates default configuration of dense layer - no constraints and both weights and bias are initialized with zeroes.
layerDense :: Num a => Int -> LayerConfiguration (Dense a)
layerDense = layerDenseWith (Initializer zeroes, []) (Initializer zeroes, [])
