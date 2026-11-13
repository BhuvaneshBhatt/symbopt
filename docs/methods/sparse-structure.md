# Sparsity and symmetry

Polynomial and SDP formulations can grow rapidly. `symbopt` therefore exploits structure before constructing larger relaxations.

## Correlative sparsity

The variable-interaction graph connects variables that appear together in polynomial terms. Its connected components expose separable polynomial structure.

## Support-aware monomial bases

SOS basis generation starts from cached monomial exponents and prunes terms outside conservative support bounds. This reduces Gram dimensions while preserving the implemented certificate space.

## Gram/PSD block propagation

Correlative structure is propagated directly into the SOS representation. Disconnected components become independent Gram blocks, while connected sparse graphs are chordally extended and split into maximal clique Gram blocks when the cliques overlap. Polynomial coefficient matching allocates overlap terms globally, so the exact SOS identity enforces consistency across clique contributions.

The structural flow is

\[
\text{polynomial support}
\rightarrow
\text{correlative components}
\rightarrow
\text{Gram blocks}
\rightarrow
\text{PSD blocks}
\rightarrow
\text{SDP}.
\]

For generic sparse SDP blocks, coefficient matrices are likewise allocated across maximal chordal cliques and explicit equality rows enforce duplicated separator entries. The reconstructed numerical result remains approximate, while exact SOS claims still require the usual exact polynomial identity and PSD recovery checks.

## Symmetry

The package detects simple even/sign and pairwise permutation symmetries. Sign-symmetry reduction is applied only when it can be done safely with the known bounds.

## Further reading

See the polynomial/SOS and semidefinite references in {doc}`../reference/references` for the mathematical background.
