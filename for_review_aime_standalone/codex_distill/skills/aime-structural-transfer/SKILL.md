---
name: aime-structural-transfer
description: Solve AIME-style mathematics with structural reductions, exact counting, and targeted verification. Use for text-only competition problems involving constrained arrangements, periodic roots, geometry, modular powers, or rational recurrences.
---

# AIME Structural Transfer

## Keep the requested quantity in view

State the exact requested quantity and its domain. Prefer a representation that determines that quantity without recovering every unknown. Record case counts and equations compactly; do not repeat an established calculation in prose.

Once a derivation covers the admissible cases and determines the requested quantity, perform a targeted check and report it. A new calculation should resolve a specific remaining gap. For example, after proving a recurrence for a requested numerator-denominator sum, do not also solve for their difference. If a check contradicts a candidate, return to the first failing equation and replace the candidate instead of repeating both claims.

The cards below are conditional aids. Use only those whose hypotheses match; an ordinary direct solution is sufficient when no card applies.

## Count structures, then their realizations

For constrained grids or assignments, distinguish **which objects occupy a group** from **their order in labeled positions**. Forced sets are not forced arrangements. A fixed set of s distinct objects has s! orders in unrestricted labeled positions. Multiply these factors only when placements are independent, and justify that the encoding can be uniquely reversed.

When objects move among groups, write a transfer matrix with source totals, destination totals, and forbidden entries. Solve its constraints before enumeration. For three equally sized groups with zero diagonal, equal row and column sums s force the off-diagonal entries to alternate t and s-t. Choose the transferred subsets, then order objects inside each destination. A final group filled by complements still needs its ordering factor. Restore any initial symmetry normalization only after proving its multiplicity.

## Count periodic roots with explicit boundaries

For a nested equation such as sin(r*pi*sin(m*x))=0 with positive integers r,m, replace the outer zero condition by sin(m*x)=k/r, where -r<=k<=r. Separate nonzero interior levels, extreme levels, and the zero level.

On 0<u<2*pi*m, each nonzero level strictly between -1 and 1 has 2m roots; each extreme level has m; the zero level has 2m-1, obtained by listing u=j*pi with 1<=j<=2m-1. Endpoint inclusion and phase shifts change the boundary count: write the admissible integer-index range rather than blindly multiplying a per-period count. Distinct levels have disjoint root sets.

Check tangency separately using the derivative at a root. For this nested sine, the outer cosine equals (-1)^k and is nonzero, so only the derivative of the inner sine can vanish there.

## Factor inequality differences before drawing regions

For chained nonlinear expressions, subtract adjacent expressions and factor before expanding or plotting. A shared shift may convert the products into coordinate differences times coordinates. For example, differences in x-yz, y-zx, z-xy factor as (x-y)(z+1) and (y-z)(x+1); translating all coordinates by one exposes sign and ordering constraints.

Partition by signs before dividing. If a bounded component becomes 0<u<v<w with u+v+w=S>0, its closure has vertices (0,0,S), (0,S/2,S/2), (S/3,S/3,S/3). Obtain them from simultaneous boundary equalities and verify every vertex against all closed inequalities. Use half the norm of a cross product for planar area in three dimensions; coordinate projection alone need not preserve area. Removing boundary segments does not change area.

## Reduce modular powers before enumerating residues

Use a small modulus to eliminate impossible divisibility patterns before lifting. For cubes modulo a power of three, unit cubes are +/-1 modulo 9 and nonunit cubes are 0. A sum of three cubes divisible by 9 therefore has either no unit bases or exactly two, whose cube signs are opposite. Three units cannot work modulo 9, even if they work modulo 3.

For h>=1, cubing gives a bijection from unit residue classes modulo 3^h to residues congruent to +/-1 modulo 9 modulo 3^(h+1). To check injectivity, equal cubes force x=y modulo 3, and then v_3(x^3-y^3)=v_3(x-y)+1 for distinct unit representatives. The domain and image have equal size. Thus an admissible unit cube determines one base class modulo 3^h. Count its representatives in the actual interval; it need not determine one integer. If all three bases are divisible by 3 and the modulus is 3^n with n>=3, divide the cubes and modulus by 27 and update the ranges. For n<=2, this all-divisible case is already automatic. For a remaining modulus of 3 use x^3=x modulo 3; for modulus 1 every tuple works. Do not apply the nonsingular Hensel rule when the derivative is divisible by the prime.

## Count chord intersections through endpoint order

Starting with fixed full chords of a disk, add independently sampled full chords in continuous general position. Each new chord adds one plus its number of distinct interior intersections. Use linearity of expectation: initial regions + number of new chords + expected intersections with fixed chords + expected pairwise intersections of new chords. This rule is not valid for arbitrary isolated interior segments.

Two chords cross exactly when their endpoints alternate around the circle. If each chord joins two distinct arcs from a collection of disjoint contiguous arcs, group by its arc-pair type. Under independent, identical continuous sampling within a shared arc: identical arc pairs cross with probability 1/2; pairs sharing exactly one arc also cross with probability 1/2. For four distinct arcs, cyclic alternation decides probability 1 or 0. Weight types by their actual probabilities; equal weights require justification. Record each symmetry class once. The unrestricted crossing probability 1/3 must not be reused after restricting endpoints without conditioning.

## Certify distance minima by a shared equality point

For sums of distances, try grouping a pair with a tractable triple. XP+XQ>=PQ is sharp exactly on segment PQ. For other points A_i, a candidate O distinct from them minimizes their distance sum if unit vectors e_i=(A_i-O)/|A_i-O| sum to zero: projection gives sum |A_i-X|>=sum |A_i-O| for every X. Three such unit vectors at 120 degrees balance; opposite vectors balance in pairs.

Construct O from the geometry, then check both the vector balance and every equality condition, including membership in the relevant segment. Adding separately sharp bounds is insufficient unless one point attains all of them. Do not extrapolate the three-point 120-degree rule to arbitrary five-point configurations.

## Exploit equal-distance geometry before elimination

Subtract squared-distance equations to obtain projections or linear loci. With A=(0,0), B=(c,0), C=(0,b), and AK=BK=AL=CL=r, write K=r(cos u,sin u), L=r(cos v,sin v). Then c=2r*cos u and b=2r*sin v. The chord KL determines |v-u|. Use interiority and the given lengths to choose the angular order. The hypotenuse and area may depend on the same sum-to-product expression, so individual angles need not be solved. Justify any area partition or signed-area orientation.

For curves rotated about the origin, inspect invariants before eliminating to a quartic. If the original curve is r_squared=q(y) with quadratic q, rotation preserves r_squared=x^2+y^2, and q(y)=q(Y) factors into two linear conditions in y,Y. Check both branches.

## Track reduced rational quantities exactly

For x=a/b in lowest terms, derive the next unreduced pair (A,B), prove g=gcd(A,B), then update the requested reduced quantity using (A/g,B/g). Modular division by g requires an invertible residue or an exact integer argument; do not reduce first and cancel blindly.

Look for a closed recurrence for the requested sum or product. For instance, a^2-ab+b^2+3ab=(a+b)^2 can close the sum after the cancellation factor is established. Prove the invariant that makes the cancellation rule persist. Split a composite modulus into coprime prime powers: use exponent reduction only in unit components and handle nonunits by divisibility.

For alternating sequences, determine endpoint type by the difference from the starting index modulo two; do not infer it from the number of labels alone.

## Finish with a focused check

Check the step most capable of changing the answer: omitted ordering factors, endpoint indices, cancellation factors, sign branches, or attainment of a bound. Substitute into the relevant original conditions. Complete the requested reduction or integer conversion, then end with exactly <answer>\boxed{N}</answer>, with 0<=N<=999.
