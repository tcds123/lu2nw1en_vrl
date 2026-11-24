---
name: aime-transfer-final
description: Produce compact, verifiable AIME solutions using target quantities, exact case accounting, and conditional structural lemmas. Applies to text-only competition mathematics, not numerical answer lookup.
---

# AIME: a compact proof of the requested quantity

Return one finished mathematical solution: the decisive reduction, necessary case counts or equations, a targeted verification, and the requested integer. Usually 400–700 words suffice; preserve essential proof when more is needed. Use a small table for repeated cases. Present the successful route once, without exploratory dialogue, repeated problem restatements, or a second full derivation.

Identify exactly what must be computed. If that quantity already follows from proved relations, compute it directly. For example, a recurrence for the requested numerator-denominator sum does not require a formula for either term separately. If a check reveals a contradiction, repair the earliest unsupported step instead of forcing the result into the expected answer form.

Use a lemma below only when its hypotheses match. Apply it directly after checking those hypotheses; re-deriving it or constructing a more general theory is unnecessary. Ordinary direct methods remain appropriate for other problems.

## Conditional reductions

**Sets versus arrangements.** Count chosen sets separately from their arrangements in labeled positions. A forced set of s distinct objects still has s! orders when positions are unrestricted. For three equal groups of size s whose objects must move to other groups, a transfer matrix with zero diagonal and row/column sums s has rows (0,t,s-t), (s-t,0,t), (t,s-t,0). Thus subset choices contribute binom(s,t)^3 for 0<=t<=s. Then count within-group orders, including any final groups fixed by complements. Check unique reconstruction; “forced contents” does not mean “one filling.”

**Roots and boundaries.** Translate nested trigonometric zeros into admissible levels. On the open interval 0<u<2*pi*m, for positive integer m, sin(u)=c has 2m roots when 0<|c|<1, m when c=+1 or -1, and 2m-1 when c=0. For the zero level, explicitly list u=j*pi. Different phases or endpoint conventions require a new integer-index range. For tangencies, evaluate the derivative at the roots rather than inferring tangency from the root count.

**Piecewise curves and sums of roots.** Bound the coordinates first, index the finitely many affine branches, and retain each branch's interval. For a sum over distinct solutions, use Vieta's root sum only for two distinct roots both admissible on that branch. Count a double root once and deduplicate shared branch endpoints. Evaluate individual roots when membership or multiplicity is uncertain. A table of branch index ranges, valid-root counts, and contributions can replace a long list of radicals.

**Factored inequalities and alternating lengths.** Subtract neighboring expressions before expanding: differences among x-yz, y-zx, z-xy factor using shifted coordinates. Partition signs before dividing. For 0<u<v<w and u+v+w=S>0, the closure has vertices (0,0,S), (0,S/2,S/2), (S/3,S/3,S/3); verify boundaries and use a cross product for area. Separately, positive lengths satisfying r_i*r_(i+1)=K obey r_(i+2)=r_i, not necessarily r_(i+1)=r_i. Track endpoint parity relative to the starting index. If they alternate a,b, use ab=K and the requested sum a+b in the cosine law.

**Cubes modulo powers of three.** For modulus divisible by 9, three cubes summing to zero have either all bases divisible by 3 or exactly two unit bases with opposite cube signs modulo 9. For e>=2, a target congruent to +/-1 modulo 9 has exactly one cube-root class modulo 3^(e-1) solving the congruence modulo 3^e; count representatives in the actual interval. In the all-divisible case, substitute a=3u, b=3v, c=3w explicitly: range 1..L becomes 1..floor(L/3), and modulus 3^e becomes 3^(e-3), not 3^(e-2). If e<=3 that case is already automatic. At modulus 3, use x^3=x. Keep range and modulus in separate columns; do not infer a one-parameter recurrence when they change differently.

**Reduced rational recurrences.** Prove the cancellation factor before tracking a reduced sum. If a,b are positive coprime integers with 3|(a+b), then gcd(a^2-ab+b^2,3ab)=3. Consequently, under x'=(x+1/x-1)/3, the reduced sum S=a+b satisfies S'=S^2/3, and the condition persists. With T=S/3, T'=T^2. For other recurrences derive their own gcd rule. Use exponent reduction only modulo coprime factors; handle nonunits by divisibility before combining residues.

**Chord intersection counts.** With fixed full chords and new independent continuous random full chords in general position, each new chord adds one plus its distinct interior intersections. Use linearity of expectation. For chords each joining two distinct disjoint contiguous arcs, independent identical continuous sampling within each arc gives crossing probability 1/2 for identical arc pairs or pairs sharing one arc; four distinct arcs give 1 exactly for cyclic alternation, otherwise 0. Weight each type once.

**Geometry certificates.** For distance minimization, XP+XQ>=PQ is sharp on segment PQ. A candidate O distinct from A_i minimizes sum XA_i if the unit vectors from O to A_i sum to zero. Try structurally defined candidates, such as a perpendicular foot onto PQ, then check vector balance and a common equality point before adding bounds. Repeated numerical optimization is not an exact certificate. In cyclic geometry, first record cyclic order and use inscribed angles, parallelism, or the exterior-secant half-difference of intercepted arcs.

## Conclude

Check the vulnerable step: multiplicity, boundary index, sign, cancellation, or simultaneous equality. Complete the requested integer conversion. End with exactly <answer>\boxed{N}</answer>, where 0<=N<=999.
