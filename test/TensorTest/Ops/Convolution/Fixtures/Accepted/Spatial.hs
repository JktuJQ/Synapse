{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Spatial where

import qualified Synapse.Tensor as T

one :: T.Tensor '[1, 3, 3] Int
one = T.conv1d @'[1] @'[0, 0] @'[1] @1 (T.ones @'[1, 2, 4]) (T.ones @'[3, 2, 2])

three :: T.Tensor '[1, 3, 2, 2, 2] Int
three = T.conv3d @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] @1 (T.ones @'[1, 2, 2, 2, 2]) (T.ones @'[3, 2, 1, 1, 1])

transposed :: T.Tensor '[1, 2, 5, 5] Double
transposed = T.convTranspose2d @'[2, 2] @'[0, 0, 0, 0] @'[1, 1] @1 @'[1, 1] (T.ones @'[1, 3, 2, 2]) (T.ones @'[3, 2, 2, 2])

pooled :: T.Tensor '[1, 2, 2, 2] Double
pooled = T.avgPool2d @'[2, 2] @'[2, 2] @'[0, 0, 0, 0] False (T.ones @'[1, 2, 4, 4])

pool1 :: T.Tensor '[1, 2, 2] Double
pool1 = T.maxPool1d @'[2] @'[2] @'[0, 0] @'[1] (T.ones @'[1, 2, 4])

pool3 :: T.Tensor '[1, 2, 2, 2, 2] Double
pool3 = T.avgPool3d @'[2, 2, 2] @'[2, 2, 2] @'[0, 0, 0, 0, 0, 0] False (T.ones @'[1, 2, 4, 4, 4])

patch1 :: T.Tensor '[1, 2, 2] Double
patch1 = T.extractPatches1d @'[2] @'[1] @'[0, 0] @'[1] (T.ones @'[1, 1, 3])

patch3 :: T.Tensor '[1, 4, 2] Double
patch3 = T.extractPatches3d @'[2, 1, 2] @'[1, 1, 1] @'[0, 0, 0, 0, 0, 0] @'[1, 1, 1] (T.ones @'[1, 1, 2, 2, 2])

transpose1 :: T.Tensor '[1, 2, 9] Double
transpose1 = T.convTranspose1d @'[3] @'[1, 0] @'[1] @1 @'[2] (T.ones @'[1, 3, 3]) (T.ones @'[3, 2, 2])

transpose3 :: T.Tensor '[1, 2, 4, 2, 5] Double
transpose3 = T.convTranspose3d @'[2, 1, 3] @'[1, 0, 0, 0, 2, 0] @'[1, 1, 1] @1 @'[1, 0, 2] (T.ones @'[1, 1, 2, 2, 2]) (T.ones @'[1, 2, 2, 1, 2])

grouped :: T.Tensor '[1, 6, 3] Double
grouped = T.conv1d @'[2] @'[1, 1] @'[2] @2 (T.ones @'[1, 4, 5]) (T.ones @'[6, 2, 2])

groupedTranspose :: T.Tensor '[1, 6, 5] Double
groupedTranspose = T.convTranspose1d @'[2] @'[1, 0] @'[2] @2 @'[1] (T.ones @'[1, 4, 2]) (T.ones @'[4, 3, 2])
