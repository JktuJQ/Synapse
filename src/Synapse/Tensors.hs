{- | Module that provides mathematical base for neural networks.

This module implements 'Synapse.Tensors.Vec.Vec' and 'Synapse.Tensors.Mat.Mat' datatypes and provides
several useful function to work with them.

Most of typeclasses of this module are working with 'DType' type family.
That is to permit instances on types that are not exactly containers, but rather wrappers of containers,
and it allows imposing additional constraints on inner type.
The best example is 'Synapse.Autograd.Symbol' from "Synapse.Autograd".
-}


{- 'ConstrainedClassMethods', 'FlexibleContexts', 'TypeFamilies', 
are needed to define 'DType' typefamily, 'Indexable', 'ElementwiseScalarOps', 'SingletonOps', 'VecOps', 'MatOps' typeclasses.
-}
{-# LANGUAGE ConstrainedClassMethods #-}
{-# LANGUAGE FlexibleContexts        #-}
{-# LANGUAGE TypeFamilies            #-}  


module Synapse.Tensors
    ( -- * 'DType' type family
      
      DType
        
      -- * 'Indexable' typeclass

    , Indexable (Index, unsafeIndex, (!), (!?))

      -- * Container-scalar operations
    , ElementwiseScalarOps ((+.), (-.), (*.), (/.), (**.), elementsMin, elementsMax)
    , SingletonOps (singleton, isSingleton, unSingleton, extendSingleton, elementsSum, elementsProduct, mean, norm)

      -- * Specific container operations
    , VecOps (dot)
    , MatOps (transpose, addMatRow, matMul)
    ) where


import Data.Kind (Type)


-- | 'DType' type family allows to get type of element for any container of that type - even for nested ones!
type family DType f :: Type


infixl 9 !, !?
-- | 'Indexable' typeclass provides indexing interface for datatypes. 
class Indexable f where
    -- | Type of index for 'Indexable' container.
    type Index f :: Type

    -- | Unsafe indexing.
    unsafeIndex :: f -> Index f -> DType f

    -- | Indexing with bounds checking.
    (!) :: f -> Index f -> DType f

    -- | Safe indexing.
    (!?) :: f -> Index f -> Maybe (DType f)


infixl 6 +., -.
infixl 7 *., /.
infixr 8 **.
{- | 'ElementwiseScalarOps' typeclass allows containers over numerical values easily work with scalars by using elementwise operations.

This typeclass operates on 'DType' to permit instances on types that are not exactly containers, but rather wrappers of containers.
The best example is 'Synapse.Autograd.Symbol' from "Synapse.Autograd".
-}
class ElementwiseScalarOps f where
    -- | Adds given value to every element of the container.
    (+.) :: Num (DType f) => f -> DType f -> f
    -- | Subtracts given value from every element of the functor.
    (-.) :: Num (DType f) => f -> DType f -> f
    -- | Multiplies every element of the functor by given value.
    (*.) :: Num (DType f) => f -> DType f -> f
    -- | Divides every element of the functor by given value.
    (/.) :: Fractional (DType f) => f -> DType f -> f
    -- | Exponentiates every element of the functor by given value.
    (**.) :: Floating (DType f) => f -> DType f -> f

    -- | Applies 'min' operation with given value to every element.
    elementsMin :: Ord (DType f) => f -> DType f -> f
    -- | Applies 'max' operation with given value to every element.
    elementsMax :: Ord (DType f) => f -> DType f -> f

{- | 'SingletonOps' typeclass provides operations that relate to singleton containers (scalars that are wrapped in said container).

All functions of that typeclass must return singletons (scalars that are wrapped in container).

This typeclass operates on 'DType' to permit instances on types that are not exactly containers, but rather wrappers of containers.
The best example is 'Synapse.Autograd.Symbol' from "Synapse.Autograd".
-}
class SingletonOps f where
    -- | Initializes singleton container.
    singleton :: DType f -> f
    -- | Return true if container represents a singleton.
    isSingleton :: f -> Bool
    -- | Unwraps singleton container.
    unSingleton :: f -> DType f

    -- Extends singleton to needed size using second argument as a reference.
    extendSingleton :: f -> f -> f

    -- | Sums all elements of container.
    elementsSum :: Num (DType f) => f -> f
    -- | Multiplies all elements of container ('Fractional' constraint is needed for efficient gradient calculation, although it may be overly restrictive in some situations).
    elementsProduct :: Fractional (DType f) => f -> f
    
    -- | Calculates the mean of all elements of container.
    mean :: Fractional (DType f) => f -> f

    -- | Calculates the Frobenius norm of all elements of container.
    norm :: Floating (DType f) => f -> f


{- | 'VecOps' typeclass provides vector-specific operations.

This typeclass operates on 'DType' to permit instances on types that are not exactly containers, but rather wrappers of containers.
The best example is 'Synapse.Autograd.Symbol' from "Synapse.Autograd".
-}
class VecOps f where
    -- | Calculates dot product of two vectors.
    dot :: Num (DType f) => f -> f -> f

{- | 'MatOps' typeclass provides matrix-specific operations.

This typeclass operates on 'DType' to permit instances on types that are not exactly containers, but rather wrappers of containers.
The best example is 'Synapse.Autograd.Symbol' from "Synapse.Autograd".
-}
class MatOps f where
    -- | Transposes matrix.
    transpose :: f -> f

    -- | Add matrix that represents row to every row of given matrix.
    addMatRow :: Num (DType f) => f -> f -> f

    -- | Mutiplies two matrices.
    matMul :: Num (DType f) => f -> f -> f
