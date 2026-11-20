{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

-- | Generator initialization, random sampling, and Philox rounds.
module Synapse.Tensor.Ops.Raw.Random
    ( -- * Generator state
      StagedGenerator
    , generatorFromSeed
    , initializeGenerator

      -- * Sampling actions
    , randomWords
    , uniform
    , bernoulli

      -- * Counter transformation
    , philox
    ) where

import Control.Monad.Trans.State.Strict (State, state)
import Data.Bits (shiftR)
import Data.Proxy (Proxy (..))
import Data.Word (Word32, Word64)
import GHC.Generics (Generic)
import System.Random (RandomGen, genWord64)

import Synapse.Tensor.Data (fromList)
import Synapse.Tensor.Ops.Conversion (FromIntegral)
import Synapse.Tensor.Ops.Raw.Construction (element)
import Synapse.Tensor.Representation (DType, Tensor (..), TensorElement (..))
import Synapse.Tensor.Shapes.Backend (backendShape)
import Synapse.Tensor.Shapes.Dimensions (Shape, shapeSize)
import Synapse.Tensor.Structure (TensorStructure, WithTensor)

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Data.Bits as B

import qualified Synapse.Tensor.Ops.Conversion as T (fromIntegral)

{- | Explicit state for random tensor computations.
Each draw produces samples and an updated state for the next draw.
Empty tensors leave the state unchanged.

The representation parameter supports staged and materialized generators through 'WithTensor'.
Generators can also be fields of user-defined tensor records passed to prepared computations.
This generator is not suitable for cryptographic use.
-}
data StagedGenerator t = StagedGenerator (t '[2] Word32) (t '[] Word64)
    deriving stock (Generic)

deriving stock instance (Show (t '[2] Word32), Show (t '[] Word64)) => Show (StagedGenerator t)

instance TensorStructure (StagedGenerator Tensor) where
    type WithTensor t (StagedGenerator Tensor) = StagedGenerator t

-- | Initialize a generator from a 64-bit seed.
generatorFromSeed
    :: Word64
    -> StagedGenerator Tensor
generatorFromSeed seed = StagedGenerator (fromList [fromIntegral seed, fromIntegral $ seed `shiftR` 32]) (fromList [0])

-- | Initialize a generator from a 64-bit seed drawn from a host random generator.
initializeGenerator
    :: RandomGen g
    => g
    -> (StagedGenerator Tensor, g)
initializeGenerator generator =
    let (seed, next) = genWord64 generator
     in (generatorFromSeed seed, next)

-- | Sample uniformly distributed 32-bit words and advance the generator.
randomWords
    :: forall sh
     . Shape sh
    => State (StagedGenerator Tensor) (Tensor sh Word32)
randomWords = state $ \(StagedGenerator (Tensor keys) (Tensor position)) ->
    let start = A.the position
        dimensions = A.constant $ backendShape $ Proxy @sh
        output = A.generate dimensions $ \ix ->
            let i = A.toIndex dimensions ix
                block = start + A.fromIntegral (i `A.quot` 4)
                (a, b, c, d) =
                    philox
                        (TensorElement $ keys A.! A.index1 0, TensorElement $ keys A.! A.index1 1)
                        ( TensorElement $ A.fromIntegral block
                        , TensorElement $ A.fromIntegral $ B.shiftR block 32
                        , 0
                        , 0
                        )
                lane = i `A.rem` 4
             in A.cond (lane A.== 0) (unTensorElement a) $
                    A.cond (lane A.== 1) (unTensorElement b) $
                        A.cond (lane A.== 2) (unTensorElement c) (unTensorElement d)
        count = shapeSize @sh
        blocks = count `div` 4 + if count `mod` 4 == 0 then 0 else 1
        next = Tensor $ A.unit $ start + A.constant (fromIntegral blocks)
     in (Tensor output, StagedGenerator (Tensor keys) next)

-- | Sample uniformly from @[0, 1)@ and advance the generator.
uniform
    :: forall sh a
     . ( Shape sh
       , DType a
       , RealFloat a
       , Fractional (TensorElement a)
       , FromIntegral (TensorElement Word32) (TensorElement a)
       )
    => State (StagedGenerator Tensor) (Tensor sh a)
uniform = do
    Tensor words_ <- randomWords @sh
    let bits = min 32 $ floatDigits (0 :: a)
        denominator = element $ 2 ^ bits
        convert word =
            unTensorElement $
                T.fromIntegral (TensorElement $ B.shiftR word $ A.constant $ 32 - bits) / denominator
    pure $ Tensor $ A.map convert words_

-- | Sample independent boolean values with the supplied probability of being true.
bernoulli
    :: forall sh
     . Shape sh
    => Double
    -> State (StagedGenerator Tensor) (Tensor sh Bool)
bernoulli probability = do
    Tensor words_ <- randomWords @sh
    let selected word = (A.fromIntegral word :: A.Exp Double) / 4294967296 A.< A.constant probability
    pure $ Tensor $ A.map selected words_

-- | Compute four pseudorandom 32-bit words from a two-word key and a four-word counter using Philox4x32-10.
philox
    :: (TensorElement Word32, TensorElement Word32)
    -> (TensorElement Word32, TensorElement Word32, TensorElement Word32, TensorElement Word32)
    -> (TensorElement Word32, TensorElement Word32, TensorElement Word32, TensorElement Word32)
philox (key0, key1) = go 10 key0 key1
  where
    xorWords
        :: TensorElement Word32
        -> TensorElement Word32
        -> TensorElement Word32
    xorWords (TensorElement left) (TensorElement right) = TensorElement $ B.xor left right

    multiply
        :: TensorElement Word32
        -> TensorElement Word32
        -> (TensorElement Word32, TensorElement Word32)
    multiply multiplier value =
        let product_ = T.fromIntegral multiplier * T.fromIntegral value :: TensorElement Word64
            high = TensorElement $ B.shiftR (unTensorElement product_) 32
         in (T.fromIntegral high, T.fromIntegral product_)

    go
        :: Int
        -> TensorElement Word32
        -> TensorElement Word32
        -> (TensorElement Word32, TensorElement Word32, TensorElement Word32, TensorElement Word32)
        -> (TensorElement Word32, TensorElement Word32, TensorElement Word32, TensorElement Word32)
    go 0 _ _ counter = counter
    go n k0 k1 (c0, c1, c2, c3) =
        let (hi0, lo0) = multiply 0xd2511f53 c0
            (hi1, lo1) = multiply 0xcd9e8d57 c2
         in go
                (n - 1)
                (k0 + 0x9e3779b9)
                (k1 + 0xbb67ae85)
                (xorWords (xorWords hi1 c1) k0, lo1, xorWords (xorWords hi0 c3) k1, lo0)
