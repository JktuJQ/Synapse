{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}

-- | Reproducible random tensor computations with explicit generator state.
-- Pass the returned generator to the next sampling operation to continue its stream.
-- Reusing a generator repeats the same samples; executing a graph does not mutate it.
-- Sampling takes place in the tensor computation and requires no host-side effects.
--
-- For a fixed library version and backend, the same seed and sequence of sampling
-- operations, including their shapes and element types, produce the same results.
-- Exact sequences and the generator's internal representation may change between versions.
-- Drawing several smaller tensors need not produce the same samples as drawing one larger tensor.
--
-- A generator can be materialized and passed to a prepared computation:
--
-- @
-- let initial = T.run T.CPU $ T.generatorFromSeed 42
--     sample = T.run1 T.CPU $ T.uniform \@'[128] \@Double
--     (values, next) = sample initial
--     (moreValues, nextAgain) = sample next
-- @
--
-- The example assumes a qualified import of this API through "Synapse.Tensor".
module Synapse.Tensor.Random
  ( Generator,
    generatorFromSeed,
    initializeGenerator,
    randomWords,
    uniform,
    bernoulli,
  )
where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.Data.Bits as B
import Data.Bits (shiftR)
import Data.Proxy (Proxy (..))
import Data.Word (Word32, Word64)
import GHC.Generics (Generic)
import Synapse.Tensor.Conversion (fromList)
import Synapse.Tensor.Shapes.Dimensions (Shape, backendShape, shapeSize)
import Synapse.Tensor.Type (DType, FromIntegral, Tensor (..), TensorElement (..), WithTensor, element)
import qualified Synapse.Tensor.Type as T
import System.Random (RandomGen, genWord64)

-- | Explicit state for random tensor computations.
-- Sampling functions return both the samples and the state for the next draw.
-- Empty tensors leave the state unchanged.
--
-- The representation parameter supports staged and materialized generators through 'WithTensor'.
-- Generators can also be fields of user-defined tensor records passed to prepared computations.
-- The constructor and storage format are private.
-- This generator is not suitable for cryptographic use.
data Generator t = Generator (t '[2] Word32) (t '[] Word64)
  deriving (Generic)

type instance WithTensor t (Generator Tensor) = Generator t

-- | Create a generator from a reproducible 64-bit seed.
-- Use the returned state with any sampling function in this module.
generatorFromSeed :: Word64 -> Generator Tensor
generatorFromSeed seed = Generator (fromList [fromIntegral seed, fromIntegral $ seed `shiftR` 32]) (fromList [0])

-- | Create a staged generator from a
-- <https://hackage.haskell.org/package/random/docs/System-Random.html host random generator>.
-- Consumes one 64-bit seed and returns the updated host generator.
-- This initializes a separate stream; it does not reproduce the host generator's samples.
initializeGenerator :: (RandomGen g) => g -> (Generator Tensor, g)
initializeGenerator generator =
  let (seed, next) = genWord64 generator
   in (generatorFromSeed seed, next)

-- | Philox4x32-10 from <https://github.com/DEShawResearch/random123 Random123>,
-- with a two-word key and a four-word counter.
philox :: (A.Exp Word32, A.Exp Word32) -> (A.Exp Word32, A.Exp Word32, A.Exp Word32, A.Exp Word32) -> (A.Exp Word32, A.Exp Word32, A.Exp Word32, A.Exp Word32)
philox (key0, key1) = go (10 :: Int) key0 key1
  where
    multiply :: A.Exp Word32 -> A.Exp Word32 -> (A.Exp Word32, A.Exp Word32)
    multiply multiplier value = let p = A.fromIntegral multiplier * A.fromIntegral value :: A.Exp Word64 in (A.fromIntegral $ B.shiftR p 32, A.fromIntegral p)
    go :: Int -> A.Exp Word32 -> A.Exp Word32 -> (A.Exp Word32, A.Exp Word32, A.Exp Word32, A.Exp Word32) -> (A.Exp Word32, A.Exp Word32, A.Exp Word32, A.Exp Word32)
    go 0 _ _ c = c
    go n k0 k1 (c0, c1, c2, c3) =
      let (hi0, lo0) = multiply (0xd2511f53 :: A.Exp Word32) c0
          (hi1, lo1) = multiply (0xcd9e8d57 :: A.Exp Word32) c2
       in go (n - 1) (k0 + 0x9e3779b9) (k1 + 0xbb67ae85) (B.xor (B.xor hi1 c1) k0, lo1, B.xor (B.xor hi0 c3) k1, lo0)

-- | Sample uniformly distributed 32-bit words in row-major order.
-- Returns the updated generator; an empty output leaves it unchanged.
randomWords :: forall sh. (Shape sh) => Generator Tensor -> (Tensor sh Word32, Generator Tensor)
-- Calls consume whole four-word Philox blocks and discard unused words.
-- The 64-bit block counter wraps after 2^64 blocks.
randomWords (Generator (Tensor keys) (Tensor position)) =
  let start = A.the position
      dimensions = A.constant $ backendShape $ Proxy @sh
      output = A.generate dimensions $ \ix ->
        let i = A.toIndex dimensions ix
            block = start + A.fromIntegral (i `A.quot` 4)
            (a, b, c, d) = philox (keys A.! A.index1 0, keys A.! A.index1 1) (A.fromIntegral block, A.fromIntegral $ B.shiftR block 32, 0, 0)
            lane = i `A.rem` 4
         in A.cond (lane A.== 0) a $ A.cond (lane A.== 1) b $ A.cond (lane A.== 2) c d
      count = shapeSize @sh
      blocks = count `div` 4 + if count `mod` 4 == 0 then 0 else 1
      next = Tensor $ A.unit $ start + A.constant (fromIntegral blocks)
   in (Tensor output, Generator (Tensor keys) next)

-- | Sample uniformly from @[0, 1)@ and return the updated generator.
-- Uses 24 random bits for 'Float' and 32 for 'Double'; one is never returned.
-- The returned samples are ordinary tensors without differentiation history.
uniform :: forall sh a. (Shape sh, DType a, RealFloat a, Fractional (TensorElement a), FromIntegral (TensorElement Word32) (TensorElement a)) => Generator Tensor -> (Tensor sh a, Generator Tensor)
uniform generator =
  let (Tensor words_, next) = randomWords @sh generator
      bits = min 32 $ floatDigits (0 :: a)
      denominator = element $ 2 ^ bits
      convert word = unTensorElement $ T.fromIntegral (TensorElement $ B.shiftR word $ A.constant $ 32 - bits) / denominator
   in (Tensor $ A.map convert words_, next)

-- | Sample a mask whose elements are true with the given probability.
-- The caller must supply a finite probability in @[0, 1]@.
-- Zero produces an all-false mask and one an all-true mask.
-- Both endpoints advance the generator as an ordinary draw of the same shape would.
-- Returns the mask and updated generator; an empty output leaves the state unchanged.
bernoulli :: forall sh. (Shape sh) => Double -> Generator Tensor -> (Tensor sh Bool, Generator Tensor)
bernoulli probability generator =
  let (Tensor words_, next) = randomWords @sh generator
   in (Tensor $ A.map (\word -> (A.fromIntegral word :: A.Exp Double) / 4294967296 A.< A.constant probability) words_, next)
