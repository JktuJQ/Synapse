# Synapse style guide

This is a working guide for the v2 refactor. Extend it as concrete decisions are
made during review. Apply it to the code being reviewed; avoid unrelated rewrites.

## Basis

The following principles carry over:

- Prefer readable, maintainable code over clever expressions.
- Separate mechanical formatting changes from changes in behaviour.
- Use descriptive camelCase names and conventional type names.
- Give modules explicit export lists; use explicit or qualified imports.
- Group exports by purpose, with types before functions in each section.
- Give top-level functions signatures; split long signatures into readable parts.
- Document public declarations, arguments, examples, and relevant laws.
- Prefer standard operators, simple composition, and `pure`.
- Choose strict fields deliberately and state deriving strategies explicitly.

## Formatting in Synapse

Use comma-first layout for multiline export lists, import lists, records, lists,
and tuples. Indent module export and import lists by four spaces. Group imports
as follows, with a blank line between groups: external unqualified, project
unqualified, external qualified, project qualified. Sort modules within each group.

Use Fourmolu with the repository's `fourmolu.yaml`. Run it through `nix develop`
so the editor and command-line checks use the same development environment.
During the refactor, format reviewed modules rather than rewriting unrelated files.

Keep lines at most 110 characters long, including comments and documentation.
Split complex expressions and signatures when useful.
Avoid trailing whitespace.

Prefer one sentence per line in documentation when it fits within 110 characters.
Wrap longer sentences at punctuation or meaningful clause boundaries.
Do not split every clause onto its own line, break tightly connected phrases,
or append the start of a new sentence just to fill the preceding line.

Keep `forall`, constraints, and argument types visually distinct in long
signatures. Do not replace meaningful
constraints with opaque aliases solely to shorten their appearance.

Outside tests, put every value signature on multiple lines: the name first,
then `::`, followed by leading `=>` and `->` on separate lines.
Apply this to class methods, default signatures, and local bindings as well.

Use leading arrows in multiline signatures and omit parentheses around a single
constraint. Argument Haddock comments follow their argument types (`-- ^`);
Fourmolu controls whether they fit on the same line.

Annotate inferred tensor representation binders as `{t :: TensorLike}` rather
than `{t}`. Keep them inferred so visible type applications retain their order.

## Modules and public API

- Facade modules such as `Synapse.Tensor` contain documentation and reexports,
  not implementations.
- Use Haddock sections to make a facade's exports navigable. Keep related
  operations together and order dimensional variants consistently: 1d, 2d, 3d.
- Keep implementation details out of facade exports. Explicit extension APIs
  must still provide enough access for users to implement their own instances.
- Use explicit import lists or qualified imports. Whole-module imports are
  appropriate when intentionally reexporting that whole module. A necessary
  `Prelude hiding (...)` import is also acceptable.
- Prefer established qualifiers: `T` for `Synapse.Tensor`, `AG` for
  `Synapse.Autograd`, `NN` for `Synapse.NN`, and `BP` or `BPE` for backprop.
- Enable language extensions where needed. Do not add project-wide extensions
  or suppress warnings merely to simplify one declaration.

## Definitions and naming

- Order exported function definitions as they appear in the module's export list.
- Keep adjacent instance declarations together, without blank lines between them.
- Put supporting types and helper definitions before the definitions using
  them, when dependencies allow it.
- Keep a private helper used in one place in a `let` or `where` unless there is
  a concrete reason to make it top-level. Add a local signature when it clarifies
  a contract or controls polymorphism; do not require redundant local signatures.
- Omit signatures for local functions with one argument, except when an explicit
  polymorphic signature is necessary for type inference or polymorphic recursion.
- Prefer `$` to an outer pair of application parentheses when it improves
  readability. Use ordinary composition for short pipelines; avoid elaborate
  point-free expressions.
- Use standard abstractions and operators before introducing custom ones.
  Introduce classes to express an actual shared contract, not merely to collect
  unrelated functions.
- Use descriptive domain names such as `features` and `targets`. Short names
  remain appropriate for genuinely generic code and familiar mathematical terms.
- Use concise shape variables such as `in_sh`, `out_sh`, and `batch_size` where
  they distinguish type-level dimensions from runtime values.
- Prefix layer record fields with the layer name, for example `denseWeights`
  and `denseBias`.
- Name newtype accessors for their meaning when that improves the API:
  `forward` is intentional. Do not mechanically impose an `unTypeName` accessor.
- Type synonyms may describe useful kinds and concepts; they need not merely
  specialize an existing type constructor.

## Data and evaluation

Prefer `newtype` for a single-field wrapper when its representation and evaluation
semantics fit. Use `data` when needed for alternatives, existential fields, or
different evaluation behaviour.

Use strict fields for stored counters and state when appropriate. Do not add
strictness mechanically to staged graphs, caches, or user computations. Document
important evaluation assumptions.

Avoid partial record selectors in new sum types. Existing shared-field records
should be reviewed on their merits rather than mechanically split into wrappers.

## Documentation

- Write public documentation in English, as library documentation rather than
  notes about the conversation or the refactoring process.
- Start with what the operation does. Include a formula, shapes, argument roles,
  and failure conditions where needed to use it correctly.
- In Tensor modules, keep function and module comments to one or two descriptive
  lines. Describe the calculation without repeating backend, staging, or
  differentiation machinery already implied by the module.
- Explain observable behaviour and restrictions. Keep implementation commentary
  in ordinary comments near the implementation.
- Link Haskell declarations with `'name'`, modules with `"Module.Name"`, and
  external resources with `<https://example.org descriptive label>`.
- Use `@code@` for inline expressions and Haddock code blocks or `>>>` examples.
  Haddock is not Markdown; do not use Markdown links or backtick fences in it.
- Both `-- |` comments and `{- | ... -}` comments are acceptable. Preserve
  readable paragraphs and consistent local formatting.
- Document unsafe operations' caller obligations explicitly. Examples should
  stay concise and avoid incidental validation or boilerplate.

## Verification

Run the applicable formatting checks and HLint alongside tests whenever tests are run.
Use `fourmolu --mode check` for reviewed modules. Select tests relevant
to a change; run the full suite for cross-cutting refactors and before release.
Do not add tests that merely mirror mechanical edits.

Check Haddock output after changes to documentation structure or public exports.
Review compilation warnings rather than hiding them. Keep formatting changes,
API changes, and behaviour changes distinguishable during review.
