module TensorTest.Compiler (accepts, rejects) where

import Data.List (isInfixOf)
import qualified Hedgehog as H
import System.Exit (ExitCode (..))
import System.Process (readProcessWithExitCode)

compile :: FilePath -> IO (ExitCode, String, String)
compile file =
  readProcessWithExitCode
    "cabal"
    [ "exec",
      "--offline",
      "--",
      "ghc",
      "-fno-code",
      "-fforce-recomp",
      "-v0",
      "-package",
      "synapse",
      file
    ]
    ""

accepts :: FilePath -> H.Property
accepts file = H.property $ do
  (status, output, diagnostic) <- H.evalIO (compile file)
  H.annotate (output ++ diagnostic)
  status H.=== ExitSuccess

rejects :: FilePath -> String -> H.Property
rejects file expected = H.property $ do
  (status, output, diagnostic) <- H.evalIO (compile file)
  H.annotate (output ++ diagnostic)
  H.assert (status /= ExitSuccess)
  H.assert (expected `isInfixOf` diagnostic)
