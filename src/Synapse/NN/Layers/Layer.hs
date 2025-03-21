{- | This module provides nessesary abstraction over layers of neural networks.

'AbstractLayer' typeclass defines interface of all layers of neural network model.
Its implementation is probably the most low-leveled abstraction of the "Synapse" library.
Notes on how to correctly implement that typeclass are in the docs for it.

'Layer' is the existential datatype that wraps any 'AbstractLayer' instance.
That is the building block of any neural network.
-}

-- 'TypeFamilies' are needed to instantiate 'DType'.
{-# LANGUAGE TypeFamilies #-}

-- 'ExistentialQuantification' is needed to define 'Layer' datatype.
{-# LANGUAGE ExistentialQuantification #-}


module Synapse.NN.Layers.Layer
    ( -- * 'AbstractLayer' typeclass

      AbstractLayer (inputSize, outputSize, nParameters, getParameters, updateParameters, symbolicForward)
    , forward

      -- * 'Layer' existential datatype

    , Layer (Layer)

      -- * 'LayerConfiguration' type alias
    , LayerConfiguration
    ) where


import Synapse.Tensors (DType)
import Synapse.Tensors.Mat (Mat)

import Synapse.Autograd (Symbolic, SymbolIdentifier, Symbol(unSymbol), SymbolMat, constSymbol)


{- | 'AbstractLayer' typeclass defines basic interface of all layers of neural network model.

Every layer should be able to pass 'Synapse.Autograd.SymbolMat' through itself to produce new 'Synapse.Autograd.SymbolMat'
(make prediction based on its parameters) using 'symbolicForward' function,
which allows for gradients to be calculated after predictions, which in turn makes training possible.

'nParameters', 'getParameters' and 'updateParameters' functions allow training of parameters of the layer.
Their implementations should match - that is 'getParameters' function should return list of length 'nParameters'
and 'updateParameters' should expect a list of the same length with the matrices in the same order as were they in 'getParameters'.

"Synapse" manages gradients and parameters for layers with erased type information using prefix system.
'Synapse.Autograd.SymbolIdentifier' is a prefix for name of symbolic parameters that are used in calculation.
Every used parameter should have unique name to be recognised by the autograd -
it must start with given prefix and end with the numerical index of said parameter.
For example 3rd layer with 2 parameters (weights and bias) should
name its weights symbol \"ml3w1\" and name its bias symbol \"ml3w2\" (\"ml3w\" prefix will be supplied).

Important: this typeclass correct implementation is very important (as it is the \'heart\' of "Synapse" library)
for work of the neural network and training, read the docs thoroughly to ensure that all the invariants are met.
-}
class AbstractLayer l where
    -- | Returns the size of the input that is supported for 'forward' and 'symbolicForward' functions. 'Nothing' means size independence (activation functions are the example).
    inputSize :: l a -> Maybe Int
    -- | Returns the size of the output that is supported for 'forward' and 'symbolicForward' functions. 'Nothing' means size independence (activation functions are the example).
    outputSize :: l a -> Maybe Int
    -- | Returns the number of parameters of this layer.
    nParameters :: l a -> Int
    
    -- | Returns a list of all parameters (those must be of the exact same order as they are named by the layer (check 'symbolicForward' docs)).
    getParameters :: SymbolIdentifier -> l a -> [SymbolMat a]
    -- | Updates parameters based on supplied list (length of that list, the order and the form of parameters is EXACTLY the same as those from 'getParameters')
    updateParameters :: l a -> [Mat a] -> l a

    {- | Passes symbolic matrix through to produce new symbolic matrix, while retaining gradients graph.
    Second matrix is a result of application of regularizer on a layer.
    -}
    symbolicForward :: (Symbolic a, Floating a, Ord a) => SymbolIdentifier -> SymbolMat a -> l a -> (SymbolMat a, SymbolMat a)

-- | Passes matrix through to produce new matrix.
forward :: (AbstractLayer l, Symbolic a, Floating a, Ord a) => Mat a -> l a -> Mat a
forward input = unSymbol . fst . symbolicForward mempty (constSymbol input)


-- | 'Layer' existential datatype wraps anything that implements 'AbstractLayer'.
data Layer a = forall l. (AbstractLayer l) => Layer (l a)

type instance DType (Layer a) = a

instance AbstractLayer Layer where
    inputSize (Layer l) = inputSize l
    outputSize (Layer l) = outputSize l

    nParameters (Layer l) = nParameters l
    getParameters prefix (Layer l) = getParameters prefix l
    updateParameters (Layer l) = Layer . updateParameters l

    symbolicForward prefix input (Layer l) = symbolicForward prefix input l


-- | 'LayerConfiguration' type alias represents functions that are able to build layers.
type LayerConfiguration l
    =  Int  -- ^ Output size of previous layer.
    -> l    -- ^ Resulting layer.
