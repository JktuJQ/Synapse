{- | This module provides nessesary abstraction over layers of neural networks.

@AbstractLayer@ typeclass defines interface of all layers of neural network model.
Its implementation is probably the most low-leveled abstraction of the @Synapse@ library.
Notes on how to correctly implement that typeclass are in the docs for it.

@Layer@ is the existential datatype that wraps any @AbstractLayer@ instance.
That is the building block of any neural network.
-}

{- @ConstrainedClassMethods@, @FlexibleContexts@, @TypeFamilies@, @TypeOperators@
are needed to instantiate @DType@.
-}

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

{- @ExistentialQuantification@ is needed to define @Layer@ datatype.
-}

{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE MultiParamTypeClasses #-}


module Synapse.NN.Layers.Layer
    ( -- * @AbstractLayer@ and @AbstractLayerM@ typeclasses

      AbstractLayer (inputSize, outputSize, nParameters, getParameters, updateParameters, symbolicForward)
    , forward

      -- * @Layer@ existential datatype

    , Layer (Layer)

      -- * @LayerConfiguration@ type alias
    , LayerConfiguration
    ) where


import Synapse.Tensors (DType)
import Synapse.Tensors.Mat (Mat)

import Synapse.Autograd (Symbolic, Symbol(unSymbol), SymbolMat, constSymbol)


{- | @AbstractLayer@ typeclass defines basic interface of all layers of neural network model.

Every layer should be able to pass @Mat@ through itself to produce new @Mat@ (make prediction based on its parameters)
using @symbolicForward@ function, which allows for gradients to be calculated after predictions,
which in turn makes training possible.

@getParameters@ function is needed to obtain information about number and and form of parameters on a layer.
That allows optimiser to work with any layer that provides such information.
@updateParameters@ function is given a list of changed parameters
(length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
and layer should change itself accordingly.

Note: any model is an @AbstractLayer@ instance too, so don't be confused by the docs on some methods.

Important: this typeclass correct implementation is very important for work of the neural network and training,
read the docs thoroughly to ensure that all the invariants are met.
-}
class AbstractLayer l where
    -- | Returns the size of the input for @forward@ and @symbolicForward@ functions that is supported. @Nothing@ means size independence (activation functions are the example).
    inputSize :: l a -> Maybe Int
    -- | Returns the size of the output of @forward@ and @symbolicForward@. @Nothing@ means size independence (activation functions are the example).
    outputSize :: l a -> Maybe Int

    -- | Returns the number of parameters of this layer.
    nParameters :: l a -> Int
    -- | Returns a list of all parameters (those must be of the exact same order as they are named (check @symbolicForward@ docs)).
    getParameters :: String -> l a -> [SymbolMat a]
    -- | Updates parameters based on supplied list (length of that list, the order and the form of parameters is EXACTLY the same as those from @getParameters@)
    updateParameters :: l a -> [Mat a] -> l a

    {- | Passes symbolic matrix through to produce new symbolic matrix, while retaining gradients graph.
    Second matrix is a result of application of regularizer on a layer.

    Given @String@ is a prefix for name of symbolic parameters that are used in calculation.
    Every used parameter should have unique name to be recognised by the autograd - 
    it must start with given prefix and end with the numerical index of said parameter.
    For example 3rd layer with 2 parameters (weights and bias) should
    name its weights symbol "l3w1" and name its bias symbol "l3w2" ("l3w" prefix will be supplied).
    It is also important so that the order of the parameters stays consistent even for @getParameters@ function
    (that will allow choosing correct gradients automatically in the training).
    -}
    symbolicForward :: (Symbolic a, Floating a, Ord a) => String -> SymbolMat a -> l a -> (SymbolMat a, SymbolMat a)

-- | Passes matrix through to produce new matrix.
forward :: (AbstractLayer l, Symbolic a, Floating a, Ord a) => Mat a -> l a -> Mat a
forward input = unSymbol . fst . symbolicForward ""  (constSymbol input)


-- | @Layer@ existential datatype wraps anything that implements @AbstractLayer@.
data Layer a = forall l. (AbstractLayer l) => Layer (l a)

type instance DType (Layer a) = a

instance AbstractLayer Layer where
    inputSize (Layer l) = inputSize l
    outputSize (Layer l) = outputSize l

    nParameters (Layer l) = nParameters l
    getParameters prefix (Layer l) = getParameters prefix l
    updateParameters (Layer l) = Layer . updateParameters l

    symbolicForward prefix input (Layer l) = symbolicForward prefix input l


-- | @LayerConfiguration@ type alias represents functions that are able to build layers.
type LayerConfiguration l
    =  Int  -- ^ Output size of previous layer.
    -> l    -- ^ Resulting layer.
