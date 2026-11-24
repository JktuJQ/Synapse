module Checksum (sha256) where

import Numeric (showHex)

import Crypto.Hash.SHA256 qualified as SHA256
import Data.ByteString qualified as BS

sha256 :: BS.ByteString -> String
sha256 = concatMap hex . BS.unpack . SHA256.hash
  where
    hex byte = case showHex byte "" of
        [digit] -> ['0', digit]
        digits -> digits
