{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Linear where

import qualified Synapse.Tensor as T

matrix :: T.Tensor '[2, 4] Int
matrix = T.matmul (T.ones @'[2, 3]) (T.ones @'[3, 4])

batched :: T.Tensor '[2, 8, 3, 5] Float
batched = T.matmul (T.ones @'[2, 8, 3, 4]) (T.ones @'[2, 8, 4, 5])

inner :: T.TensorElement Double
inner = T.dot (T.ones @'[4]) (T.ones @'[4])

empty :: T.Tensor '[2, 3] Int
empty = T.matmul (T.ones @'[2, 0]) (T.ones @'[0, 3])

expanded :: T.Tensor '[8, 2, 4] Float
expanded = T.matmul (T.broadcast @'[8, 2, 3] (T.ones @'[2, 3])) (T.ones @'[8, 3, 4])
