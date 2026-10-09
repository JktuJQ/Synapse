module TensorTest.Support (approxList) where

import Control.Monad (forM_)
import qualified Hedgehog as H

-- | Compare finite floating-point results with absolute and relative tolerance.
-- Report each failing pair rather than a single opaque boolean.
approxList :: (RealFloat a, Show a) => [a] -> [a] -> H.PropertyT IO ()
approxList actual expected = do
  length actual H.=== length expected
  forM_ (zip actual expected) $ \(x, y) -> do
    H.annotateShow (x, y)
    H.assert (not (isNaN x || isNaN y || isInfinite x || isInfinite y))
    H.assert (abs (x - y) <= 1e-5 * max 1 (abs y))
