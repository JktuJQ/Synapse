{
  description = "Synapse Haskell development environment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs =
    {
      self,
      flake-utils,
      nixpkgs,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        haskellPackages = pkgs.haskell.packages.ghc98;
      in
      {
        formatter = pkgs.nixfmt;

        devShells.default = pkgs.mkShell {
          packages = [
            haskellPackages.cabal-install
            haskellPackages.ghc
            haskellPackages.ghcid
            haskellPackages.haskell-language-server
            haskellPackages.hlint
            haskellPackages.ormolu
            pkgs.clang
            pkgs.llvm
            pkgs.libffi
            pkgs.nixfmt
            pkgs.pkg-config
          ];

          ACCELERATE_LLVM_CLANG_PATH = "${pkgs.clang}/bin/clang";

          shellHook = ''
            echo "Synapse dev shell: GHC ${haskellPackages.ghc.version}"
          '';
        };
      }
    );
}
