# matfun: theory

This note describes the mathematics behind matfun: the functions $\exp$, $\log$, $\sqrt{\cdot}$ and $x \mapsto x^\alpha$ ($\alpha > 0$) of a symmetric $3\times3$ matrix (`smf`, Sections 1–4) and of a general real $3\times3$ matrix (`gmf`, Section 5), together with their first and second derivatives.

## The difficulty

For a symmetric matrix, a matrix function is easy to define and, in exact arithmetic, easy to compute: diagonalise $A = V \mathrm{diag}(\lambda) V^\mathsf{T}$ and apply $f$ to the eigenvalues. The difficulty lies in the derivatives. A Newton solver needs $\partial F/\partial A$, a sensitivity analysis also needs the second derivative, and in a finite element computation one or both can be required at every integration point, possibly millions of times per iteration.

The derivatives can be expressed through divided differences of $f$ at the eigenvalues (Sections 3 and 4). When two eigenvalues coincide or nearly coincide, the textbook form $(f(x) - f(y))/(x - y)$ becomes $0/0$ or loses most of its digits to cancellation, although its limit $f'(x)$ is perfectly well defined. Repeated eigenvalues are not an edge case in mechanics: the undeformed state $A = I$, uniaxial and equibiaxial states, and all states close to them have them.

Differentiating through an eigensolver, as automatic differentiation of a generic symmetric eigensolver does, does not help either. The eigenvectors are not differentiable at repeated eigenvalues, and their derivatives contain factors $1/(\lambda_i - \lambda_j)$, whereas $F$ and its derivatives are smooth. The derivatives therefore have to be assembled from quantities that remain smooth, namely divided differences evaluated in a numerically stable way, with the eigenvectors entering only algebraically.

Finally, the evaluation should ideally be closed-form, free of iterations and free of data-dependent branches, so that it vectorises over many matrices on CPUs (SIMD) and GPUs.

## Choice of method

Moler and Van Loan [7] review nineteen ways to compute the matrix exponential: series methods (Methods 1–4), ODE solvers (5–7), polynomial methods (8–13, among them Cayley–Hamilton and Lagrange and Newton interpolation), matrix decompositions (14–18) and splitting (19). For symmetric matrices, Method 14 ("Eigenvectors"), $e^{A} = V e^{D} V^{-1}$, stands out: it "works well for any other symmetric matrix because the eigenvectors can be chosen orthogonal", and the resulting algorithm has "all the attributes we desire – except that it is limited to symmetric matrices" [7]. We use this method, with a closed-form eigendecomposition and $V^{-1} = V^\mathsf{T}$, for all four functions.

The derivatives bring in the divided differences of $f$ at the eigenvalues, the same quantities on which Method 10 (Newton interpolation) is built [7]. Here they enter through the Daleckii–Krein formulas for the Fréchet derivatives rather than through an interpolation polynomial for $f(A)$.

## Notation

| Symbol | Meaning |
|---|---|
| $A$ | real symmetric $3\times3$ matrix |
| $\lambda_1,\dots,\lambda_n$ | eigenvalues of $A$ |
| $V$ | orthogonal matrix of eigenvectors $v_1,\dots,v_n$, $A = V \mathrm{diag}(\lambda)\, V^\mathsf{T}$ |
| $E,\ E_1,\ E_2$ | symmetric directions (perturbations of $A$); $\tilde E = V^\mathsf{T} E V$ |
| $\circ$ | Hadamard (entrywise) product |
| $L(A,E)$, $L_2(A,E_1,E_2)$ | first and second Fréchet derivative of $F(A) = f(A)$ |
| $\mathbf v$ | Voigt vector $(A_{11}, A_{22}, A_{33}, A_{12}, A_{13}, A_{23})$, without factors of 2; $E_p$ is the symmetric basis matrix of component $p$ |
| $J,\ H$ | $J_{ij,p} = \partial F_{ij}/\partial v_p$ and $H_{ij,pq} = \partial^2 F_{ij}/\partial v_p \partial v_q$ |

## 1. Closed-form eigendecomposition

**Scaling.** Before the decomposition, we divide $A$ by the power of two $\sigma = 2^{\lfloor \log_2 \max_{ij}\lvert A_{ij}\rvert \rfloor}$, and afterwards we multiply the eigenvalues by $\sigma$. Both operations are exact in binary floating-point arithmetic, so they change no result in the normal range. Without them, the cross products below, which scale like $\lvert A\rvert^4$, overflow or underflow once the entries leave the range of about $10^{\pm 76}$, and the eigenvectors become wrong without any warning.

**A well-separated eigenvalue.** All steps work on the deviator $B = A - mI$, $m = \tfrac13 \mathrm{tr}\, A$. Its diagonal entries $A_{kk} - m$ are exact whenever $A_{kk}$ is within a factor of two of $m$ (Sterbenz), so near $A = I$ the deviator, and with it the eigenvectors and the differences $\lambda_a - m$, keep full relative accuracy even when $B$ is far below the rounding level of $A$. Because $m$ is rounded, $\mathrm{tr}\, B = 3\tau$ is not exactly zero; the trigonometric formula below needs a traceless matrix, so we apply it to $B - \tau I$ and normalize that matrix by a power of two of the size of its largest entry (both exact operations). With $B$ now denoting this traceless, normalized deviator,

$$
p = \lVert B \rVert_F^2, \qquad q = \det B,
$$

$$
r = \frac{3\sqrt6\, q}{p^{3/2}} \in [-1, 1], \qquad \varphi = \tfrac13 \arccos r,
$$

the eigenvalues of $B$ are $\sqrt{2p/3}\,\cos(\varphi + 2\pi k/3)$, $k = 0, 1, 2$; this is Smith's trigonometric formula [4], written in $p$ and $q$. We take only one eigenvalue from it, namely the one farthest from the other two,

$$
\beta = \sqrt{\tfrac{2p}{3}}\,\cos\varphi \quad (r \ge 0), \qquad
\beta = \sqrt{\tfrac{2p}{3}}\,\cos\!\left(\varphi + \tfrac{2\pi}{3}\right) \quad (r < 0),
$$

because it remains well separated even when the other two coincide. Only if the deviator vanishes do we treat the matrix as $A = mI$.

**Its eigenvector.** The eigenvector of a simple eigenvalue is orthogonal to the rows of $B - \beta I$ and can therefore be computed as the cross product of two of these rows [5]. We compute all three cross products and take the one with the largest norm, which avoids the loss of accuracy for two nearly parallel rows. Taking the rows of $A - \lambda I$ instead would subtract two numbers close to 1 near $A = I$ and lose the eigenvector there.

**The remaining eigenpairs.** We complete the eigenvector to an orthonormal basis by two unit vectors $u, w$ and diagonalise the remaining $2\times2$ problem

```math
\begin{pmatrix} u^\mathsf{T} (A - mI) u & u^\mathsf{T} (A - mI) w \\ w^\mathsf{T} (A - mI) u & w^\mathsf{T} (A - mI) w \end{pmatrix}
```

with a single Jacobi rotation by $\theta = \tfrac12 \mathrm{atan2}(2\tilde a_{12},\ \tilde a_{11} - \tilde a_{22})$ [3]. The C++ kernel obtains $\cos\theta$ and $\sin\theta$ from half-angle formulas, i.e. from square roots, instead of evaluating atan2, cos and sin. Since $V$ consists of a unit vector, its orthonormal complement and a rotation within that complement, it is orthonormal by construction, including the case of repeated eigenvalues. The rotation angle is taken from the deviator ($\tilde a_{11} - \tilde a_{22}$ of $A$ itself would cancel near $A = I$), the eigenvalues from the Rayleigh quotients of $A$ in the rotated basis, which keeps small eigenvalues of spectra over several orders of magnitude accurate. For the logarithm we also form the Rayleigh quotients $x_a$ of the deviator (Section 2).

## 2. Function value

The matrix function is

$$
F = f(A) = V \mathrm{diag}\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\, V^\mathsf{T}
$$

[2, Ch. 1]. For $\sqrt{\cdot}$ and $x^\alpha$, we set eigenvalues below zero to 0; for a positive semidefinite $A$ they can only arise from round-off. For eigenvalues near $m$, $\lvert x_a\rvert \le m/2$, the logarithm is evaluated as $\log \lambda_a = \log(\sigma m) + \mathrm{log1p}(x_a/m)$ with $x_a = \lambda_a/\sigma - m$ from the deviator: near $A = I$ its values are tiny, and $\log(\sigma(m + x_a))$ would lose them to the rounding of $m + x_a$ (relative error $10^{-5}$ of $\log A$ for $A = I + 10^{-12}E$), whereas the rounding of $m$ here cancels exactly against $A - mI$.

## 3. First derivative

By the Daleckii–Krein theorem [1], [2, Thm. 3.11], the Fréchet derivative in direction $E$ is

$$
L(A, E) = V \bigl(D \circ \tilde E\bigr) V^\mathsf{T}, \qquad D_{ab} = f[\lambda_a, \lambda_b],
$$

with the first divided difference $f[x,y] = (f(x) - f(y))/(x - y)$ and $f[x,x] = f'(x)$ [6]. Because this quotient loses digits for $x \approx y$, we evaluate it in the following forms, where $z = (x-y)/(x+y)$ and $s = (x+y)/2$, and $\mathrm{expm1}(t) = e^t - 1$ and $\mathrm{log1p}(t) = \log(1+t)$ denote the library functions that compute these expressions accurately for small $t$:

| $f$ | $f[x,y]$ |
|---|---|
| $\exp$ | $e^{\max(x,y)}\,\bigl(-\mathrm{expm1}(-d)\bigr)/d$ with $d = \lvert x - y\rvert$ |
| $\log$ | $2\mathrm{artanh}(z) / \bigl(z\,(x+y)\bigr)$ for $10^{-3} \le \lvert z\rvert < 0.5$; $\ (1 + z^2/3 + z^4/5 + z^6/7)/s$ for $\lvert z\rvert < 10^{-3}$; the quotient otherwise |
| $\sqrt{\cdot}$ | $1/(\sqrt x + \sqrt y)$ |
| $x^\alpha$ | $s^{\alpha-1}\,\bigl(\mathrm{expm1}(\alpha\mathrm{log1p} z) - \mathrm{expm1}(\alpha\mathrm{log1p}(-z))\bigr)/(2z)$ for $\lvert z\rvert < 0.5$; the quotient otherwise |

The exponential form factors out the larger exponential, so that it cannot overflow. The powers $s^{\alpha-1}$ (and $f'(x) = \alpha x^{\alpha-1}$) are evaluated with the exponent $\alpha - 1 = b_h + b_l$ split exactly (TwoSum), $s^{\alpha-1} = s^{b_h}(1 + b_l \log s)$: the rounding error of $\alpha - 1$ alone would be multiplied by $\lvert\log s\rvert$, i.e. cost two digits at $s \approx 10^{\pm100}$. The logarithmic form follows from $\log x - \log y = \log\bigl((1+z)/(1-z)\bigr) = 2\mathrm{artanh}\, z$, and its series from the Taylor series of $\mathrm{artanh}$. For the power, we write $x = s(1+z)$ and $y = s(1-z)$, so that $x^\alpha - y^\alpha = s^\alpha\bigl((1+z)^\alpha - (1-z)^\alpha\bigr)$, and evaluate both powers through $\mathrm{expm1}$ and $\mathrm{log1p}$.

**Jacobian with respect to the Voigt components.** As a fourth-order tensor, the first derivative reads $K_{ijkl} = \partial F_{ij}/\partial A_{kl} = \sum_{a,b} D_{ab}\, V_{ia} V_{ka} V_{jb} V_{lb}$. A Voigt component changes either one diagonal entry or both off-diagonal entries $A_{kl}$ and $A_{lk}$, so that

$$
J_{ij,p} = K_{ijkk} \ \ \text{for } E_p = e_k e_k^\mathsf{T}, \qquad J_{ij,p} = K_{ijkl} + K_{ijlk} \ \ \text{for } E_p = e_k e_l^\mathsf{T} + e_l e_k^\mathsf{T},\ k \ne l.
$$

The kernel evaluates this through the eigenprojectors $P_a = v_a v_a^\mathsf{T}$ and never forms $K$. For Voigt input, the library returns $J_{ij,p}$ for the six independent entries $ij$, i.e. the Jacobian of the map $\mathbf v \mapsto F$ in Voigt form. For matrix input, it returns the derivative of $A \mapsto f\bigl((A + A^\mathsf{T})/2\bigr)$, i.e. $\tfrac12 (K_{ijkl} + K_{ijlk})$, which equals $J_{ij,p}$ for a diagonal entry $kl$ and $\tfrac12 J_{ij,p}$ for an off-diagonal one. This tensor has the symmetries $ij \leftrightarrow ji$, $kl \leftrightarrow lk$ and $(ij) \leftrightarrow (kl)$. The second derivative is formed from $H$ in the same way.

## 4. Second derivative

The second Fréchet derivative is given by the second-order Daleckii–Krein formula [1],

$$
L_2(A, E_1, E_2) = V S V^\mathsf{T}, \qquad
S_{ab} = \sum_m f[\lambda_a, \lambda_m, \lambda_b]\,\bigl(\tilde E_{1,am} \tilde E_{2,mb} + \tilde E_{2,am} \tilde E_{1,mb}\bigr),
$$

and the Hessian with respect to the Voigt components is $`H_{ij,pq} = L_2(A, E_p, E_q)_{ij}`$. It contains the second divided difference [6]

$$
f[x,y,z] = \frac{f[y,z] - f[x,y]}{z - x}, \qquad f[x,x,x] = \tfrac12 f''(x).
$$

This recursion suffers from the same cancellation as the first divided difference, now when all three points are close. We therefore distinguish two cases.

If the three points do not form a cluster (defined below), we apply the recursion to the pair with the largest distance, with the stable first divided differences from Section 3.

If all three points lie in a cluster around the middle value $\mu$, we use a Taylor series. The divided difference of the monomial $t^j$ over $k+1$ points is the complete homogeneous symmetric polynomial of degree $j - k$ in these points [6, Eq. (27)]. Expanding $f$ around $\mu$ and applying this to the points $\mu + u$, $\mu$ and $\mu + w$, with $u = x_{\min} - \mu$ and $w = x_{\max} - \mu$, yields

$$
f[x,y,z] = \sum_{k \ge 0} \frac{f^{(k+2)}(\mu)}{(k+2)!}\, h_k(u, w), \qquad h_k(u,w) = \sum_{i=0}^{k} u^i w^{k-i},
$$

where the middle point drops out because it coincides with the expansion point. The coefficients are

$$
\frac{f^{(k+2)}(\mu)}{(k+2)!} = \frac{e^\mu}{(k+2)!} \ \ (\exp), \qquad
\frac{(-1)^{k+1}}{(k+2)\,\mu^{k+2}} \ \ (\log), \qquad
\binom{\alpha}{k+2}\,\mu^{\alpha-k-2} \ \ (x^\alpha),
$$

and the factors $e^\mu$ and $\mu^\alpha$ equal $f(\mu)$, which is already available. We call the points a cluster if $x_{\max} - x_{\min} \le 1$ (exp) or if $\lvert x - \mu\rvert \le \tau\mu$ for all three of them, with $\tau = 0.25$ for log and $\tau = \min(0.25,\ 3/\alpha)$ for $x^\alpha$, and we truncate the series after 30 terms.

For exp and log the coefficients do not grow with $k$, and the first neglected term is smaller than $31 \cdot 0.25^{30} < 10^{-16}$ relative to the leading one (for exp, $31/32!$). For $x^\alpha$, the binomial coefficients grow with $\alpha$: for large $\alpha$, $\binom{\alpha}{k+2}/\binom{\alpha}{2} \approx 2\alpha^k/(k+2)!$. A fixed radius therefore fails for large exponents; with $\tau = 0.25$, the error of $f[x,y,z]$ grows from $10^{-15}$ at $\alpha = 40.5$ to $10^{-8}$ at $\alpha = 60.5$ and $3\cdot10^{-3}$ at $\alpha = 100.5$. With $\tau = 3/\alpha$, the $k$-th term relative to the leading one behaves like $2(k+1)\,3^k/(k+2)!$, which is below $10^{-19}$ at $k = 30$. Up to $\alpha = 12$, the radius stays 0.25.

We checked $f[x,y,z]$ against mpmath with 50 significant digits, including points at the switch between series and recursion. The relative error is at most $4\cdot10^{-15}$ for exp, log and $x^\alpha$ with $\alpha = 0.5, 0.3, 1.7$, and below $10^{-15}$ for $x^\alpha$ with $\alpha$ up to 300.5 at the points $(0.8, 1, 1.25)$. The test `python/tests/test_mpref.py` compares $F$, $J$ and $H$ with a 60-digit reference that uses no divided differences (an mpmath eigensolver for $F$, finite differences for $J$ and $H$); for the reference cases in `testdata/cases.txt`, the relative errors are at most $3\cdot10^{-14}$, and $1.2\cdot10^{-12}$ for the two ill-conditioned matrices per function (eigenvalue ratios down to $10^{-5}$).

For large $\alpha$, the function $x^\alpha$ itself is ill-conditioned: a relative error $\delta$ in an eigenvalue becomes $\alpha\delta$ in $\lambda^\alpha$. Since the eigenvalues carry errors of a few units in the last place, errors of order $\alpha\varepsilon$ in $F$, $J$ and $H$ cannot be avoided in double precision; we measured 0.4 to 2 times $\alpha\varepsilon$ for general matrices, with $\alpha$ up to 300.5.

## 5. General (non-symmetric) matrices

`gmf` computes $F = f(A)$, $J_{abcd} = \partial F_{ab}/\partial A_{cd}$ and $H_{abcdef} = \partial^2 F_{ab}/\partial A_{cd}\partial A_{ef}$ for a general real $3\times3$ matrix, with the principal branch of $\log$, $\sqrt{\cdot}$ and $x^\alpha$. Such a matrix may have a complex pair of eigenvalues (rotations) and may be defective (simple shear $I + \gamma e_1 e_2^\mathsf{T}$ is a Jordan block), so the eigenvector approach of Sections 1–4 is not available: at a defective matrix $V$ is singular, near one it is ill-conditioned. We use instead the polynomial Methods 8–13 of [7]: $f(A)$ equals the Hermite interpolation polynomial of $f$ on the spectrum, evaluated at $A$ [2, Ch. 1], and this holds for every matrix, defective or not.

**Newton–Hermite form.** Let $\mu_1, \mu_2, \mu_3$ be the eigenvalues, $w_0 = I$, $w_1 = A - \mu_1 I$, $w_2 = (A - \mu_1 I)(A - \mu_2 I)$ and $N_j = \lbrace \mu_1, \dots, \mu_{j+1} \rbrace$. Then

$$
F = \sum_{j=0}^{2} f[N_j]\, w_j, \qquad
L(A, E) = \sum_{j,k} f[N_j \cup N_k]\, w_j E w_k, \qquad
L_2(A, E_1, E_2) = \sum_{j,k,l} f[N_j \cup N_k \cup N_l]\,\bigl(w_j E_1 w_k E_2 w_l + w_j E_2 w_k E_1 w_l\bigr),
$$

with unions of multisets, i.e. confluent divided differences $f[\mu_1^a \mu_2^b \mu_3^c]$ with $a = k + 1$ for the derivative of order $k$ [6]. The derivative formulas follow from applying the Newton form to the block matrix with $A$ on the diagonal and $E$ in the upper right block [2, Thm. 3.6],

```math
f\begin{pmatrix} A & E \\ 0 & A \end{pmatrix} = \begin{pmatrix} f(A) & L(A, E) \\ 0 & f(A) \end{pmatrix};
```

its spectrum is that of $A$ with doubled multiplicities, and the tensor-product Newton form is exact because $w_3 = \chi_A(A) = 0$. In index form, $`J_{abcd} = \sum_{jk} D_{jk} (w_j)_{ac} (w_k)_{db}`$ and $H_{abcdef} = T_{acdefb} + T_{aefcdb}$ with $`T_{acdefb} = \sum_{jkl} D_{jkl} (w_j)_{ac} (w_k)_{de} (w_l)_{fb}`$. Nineteen divided differences are needed (3 for $F$, 6 for $J$, 10 for $H$), with up to nine nodes, i.e. up to $f^{(8)}$; this cannot be avoided, since at a Jordan block of size 3 already $L$ needs $f^{(5)}$.

**Eigenvalues.** We compute the eigenvalues $x_k$ of $S = A - mI$ (traceless after removing the rounding residue $\tau$ as in Section 1, and normalized twice: by $\max\lvert S_{ij}\rvert$ and by the size of the roots) from $x^3 - Px - Q = 0$, $P = \tfrac12\mathrm{tr}\,S^2$, $Q = \det S$: the real root farthest from the others by the trigonometric formula if all three are real, else the only real root by Cardano's formula with the cancellation-free sign; the other two from the deflated quadratic, a real or a conjugate pair. Near a multiple eigenvalue these roots carry errors of order $\varepsilon^{1/2}$ or $\varepsilon^{1/3}$, but this does not matter: $F$, $L$ and $L_2$ are symmetric functions of the nodes, so they depend only on the coefficients of $\chi$, and their error is proportional to $\chi_\mu(A)$, which is $O(\varepsilon)$ (measured: $10^{-17}$ at eigenvalue errors of $7\cdot10^{-6}$).

The coefficients themselves, however, lose information in two situations, which we repair by refining the eigenvalues with $A$ itself:

- an eigenvalue far from $m$, e.g. $\lambda = 10^{-12}$ next to $1$ and $2$: $m + x$ has the absolute error $\varepsilon m$, so the small eigenvalue keeps only four digits;
- two eigenvalues close to each other but far from the third (spectrum $(10^3, 1, 0.01)$): their splitting comes from a discriminant $P - \tfrac34 x_1^2$ with $P \sim \lvert S\rvert^2$, i.e. with the error $\varepsilon\lvert S\rvert^2/\mathrm{gap}$.

If the two non-separated eigenvalues are small against the separated real one $\mu_1$, or the deviator is not dominated by the shift ($\lvert m\rvert \le \max\lvert S_{ij}\rvert$) and $\mu_1$ is separated from the pair, $\mu_1$ is refined by Newton steps on $\det(A - \mu I)$, computed by elimination with partial pivoting (backward stable, and exact for a diagonal matrix), and the pair is recomputed from its product and sum, $\pi = \det A / \mu_1$ and $\sigma = (c_2(A) - \pi)/\mu_1$ ($c_2$: sum of the principal $2\times2$ minors; for $\lvert\mu_1\rvert$ smaller than the pair $\sigma = \mathrm{tr}\,A - \mu_1$, $\pi = c_2 - \mu_1\sigma$). The pair may change its type here: for strongly non-normal matrices the roots of $S$ can show a spurious complex pair. Other single real eigenvalues far from $m$ get the same Newton steps.

One case needs more than the characteristic polynomial: a conjugate pair close to the negative real axis (log and $x^\alpha$, e.g. a rotation by an angle close to $180°$). There the principal branch jumps between the two nodes, $f[\mu_+, \mu_-] \approx 2\pi i/(\mu_+ - \mu_-)$ for log, so the result depends on the imaginary part $\beta$ of the pair itself, and the discriminant loses $\varepsilon\lVert A\rVert^2/\beta^2$ in it. If the refined pair has a negative real part, we take it from the $2\times2$ block $B = Q^\mathsf{T} S Q$ of the deviator on the invariant subspace of the pair: $Q$ is an orthonormal basis of the orthogonal complement of the left eigenvector of $\mu_1$, which is orthogonal to the columns of $A - \mu_1 I$ (their largest cross product). With $d = (b_{11} - b_{22})/2$, $\beta^2 = -(d^2 + b_{12} b_{21})$ is formed from shifted entries, without cancellation against the square of the centre, so $\beta$ has the relative error $\varepsilon\lVert A\rVert/\beta$ of the problem itself. Wherever a refined eigenvalue is used, $w$ is formed from $A - \mu I$ and node differences from the $\mu$; otherwise from $S - xI$ and the $x$, which keep the accuracy near $A = I$.

**Real basis.** For a complex pair, $w_1, w_2$ and the divided differences are complex. With the real basis $r_0 = I$, $r_1 = S - \mathrm{Re}\,x_1 I$, $r_2 = (S - \mathrm{Re}\,x_1 I)(S - \mathrm{Re}\,x_2 I) - \mathrm{Im}\,x_1 \mathrm{Im}\,x_2 I$ we have $w = T r$ with a unit lower triangular $T = I + iN$, so that $`F = \sum_m \mathrm{Re}(T^\mathsf{T} D)_m\, r_m`$, and likewise for $J$ and $H$ with $\mathrm{Re}(T^\mathsf{T} D T)$ and $\mathrm{Re}$ of $D$ contracted with $T$ on all three indices. Complex arithmetic is needed only for the 19 scalars.

**Divided differences.** The recursion $f[S \cup \lbrace u, v \rbrace] = (f[S \cup u] - f[S \cup v])/(u - v)$ compounds rounding errors over up to eight levels (measured: up to $10^{-9}$ relative for eight nodes at moderate distances), so we use residues instead. By the residue theorem,

$$
f[\mu_1^a \mu_2^b \mu_3^c] = \sum_{G} \bigl[G\bigr]\Bigl( f(t) \Big/ \prod_{y \notin G} (t - y)^{m_y} \Bigr),
$$

summed over groups $G$ of nodes, where $[G]$ is the divided difference over the nodes of $G$ only. A group is either all three nodes, if their Taylor series converges fast enough (radius $\le 0.7\,c$ around the real centre $c$ for log and $x^\alpha$, with the singularity at 0; additionally $\le 3/\alpha$ for large $\alpha$, as in Section 4; radius $\le 4$ for exp), or else the closest real or conjugate pair, if it qualifies and the pole of the third node is far enough (pole ratio $\le 0.6$), or else every node on its own. A group is evaluated by the Taylor series of $f/\prod (t-y)^{m_y}$ around its centre: $[x_0, \dots, x_n]\,g = \sum_k g_{n+k}\, h_k(x_0 - c, \dots, x_n - c)$ with the complete homogeneous symmetric polynomials $h_k$ [6]. A single node contributes the Taylor coefficient of order $m_y - 1$ of $f/\prod_{z\ne y}(t - z)^{m_z}$ at the node. Rounding errors then enter only in the final sum. For the 19 divided differences, the kernel streams over the series index $k$: the coefficients of $f/(t - y)^r$, $r \le 3$, follow by a recursion in $k$, and the $h_k$ of all 19 multisets by one update each from a parent multiset, so that no arrays of series coefficients are stored. The number of terms follows from a bound on the terms (at most about 110 in the worst case at the group boundary, about 20 near $A = I$); in a SIMD vector all lanes use the largest. In the pure-JAX version it is fixed (128), because code generators that translate the jaxpr cannot nest while loops.

The nodes are ordered so that the node with the largest high derivatives comes last (exp: the largest real part; log and $x^\alpha$: the smallest modulus, also for $\alpha > 1$, since $f^{(k)} \propto x^{\alpha-k}$ grows towards 0 for $k > \alpha$); a conjugate pair stays adjacent, and a pair close to the negative real axis comes after the real node: the large divided differences across the jump of $f$ then only multiply $w_2$, whereas with the pair first $J$ and $H$ lose $\varepsilon/\beta^k$ (relative error $2\cdot10^{-7}$ in $H$ of the log of a rotation by $179.9°$, $2\cdot10^{-13}$ with the real node first). With the order by $\lvert f'\rvert$ instead, $H$ of $x^{1.7}$ at the spectrum $(10^3, 1, 0.01)$ had an error of $2\cdot10^{-7}$; with this order it is $2\cdot10^{-13}$.

**Scaling.** For log and $x^\alpha$ we scale $A$ by a power of two as in Section 1. The logarithm uses $\log(\sigma m) + \mathrm{log1p}(x/m)$ near $m$ and $\log(\sigma\mu)$ elsewhere. For $x^\alpha$, the factor $\sigma^{\alpha-k}$ of the output of order $k$ ($F$, $J$, $H$) is folded into the coefficients as $\lvert\lambda\rvert^{\alpha-k}\mu^k$ with the eigenvalue $\lambda = \sigma\mu$ of $A$ and the exponent split exactly as in Section 3, so that neither factor over- nor underflows on its own ($(I + 10^6 e_1e_2^\mathsf{T})^{60.5}$, $10^{\pm250}$-scaled matrices). For exp we factor out $e^{m + \max \mathrm{Re}\, x}$, so that no coefficient exceeds 1 (spectra like $(700, 700, -1500)$).

**Accuracy.** Against a reference that evaluates the same formulas with 100 digits, and that agrees with mpmath's `expm`, `logm`, `sqrtm` and `powm` and finite differences to the last digit, the errors of $F$ and $J$ stay within 60 times the unavoidable error, i.e. the change of the exact result when the entries of $A$ are perturbed by a relative $\varepsilon$, over 3000 random spectra (separated, clustered, complex, with Jordan coupling, with eigenvalues from $10^{-6}$ to $10^6$, exp up to $\pm50$); for $H$ within a few hundred times, at absolute errors below $3\cdot10^{-13}$. The cases of `testdata/cases_general.txt` agree with an independent 60-digit reference (mpmath and finite differences, `python/tests/test_mpref.py`) to $2.5\cdot10^{-14}$. For symmetric input, $J$ and $H$ projected on symmetric directions equal those of `smf` to $3\cdot10^{-15}$. In a Newton method for $\exp(X) = B$, for $\log X = B$ and for a Hencky-type potential $\tfrac12\lvert\log X\rvert^2$ (tangent $J^\mathsf{T}J + \log X : H$), the iteration converges quadratically down to the level set by the conditioning of the problem, the same with gmf's tangent as with the exact one; dropping $H$ (Gauss–Newton) gives only linear convergence.

Limitations: exp of a nearly double eigenvalue at large $\lVert A\rVert$ loses a factor $\lVert A\rVert$ (the discriminant of the pair has the error $\varepsilon\lVert A\rVert^2$; $2\cdot10^{-13}$ at $\lVert A\rVert \approx 50$); powers of matrices with a departure from normality of order $10^{200}$ give NaN; for a complex pair close to the negative real axis the principal logarithm and powers are ill-conditioned themselves (condition $\sim 1/(\pi - \theta)^k$ for $F$, $J$, $H$ at a rotation by $\theta$), and the errors stay at that level (within a factor 10 of the unavoidable error up to $\theta = 179.999°$); a real eigenvalue $\le 0$ gives NaN for log, sqrt and pow.

**Cost.** With AVX-512 on one core, `gmf` takes 1.4–2.1 µs per matrix for $F$, $J$, $H$ (1.6–2.5 times `smf` in the matrix layout), 0.15–0.3 µs for $F$ alone; on an NVIDIA L40S, 33 ns per matrix for $F$, $J$, $H$ (1.6 times `smf`).

## References

1. Ju. L. Daleckii, S. G. Krein: Integration and differentiation of functions of Hermitian operators and applications to the theory of perturbations. *AMS Translations*, Ser. 2, Vol. 47 (1965), 1–30.
2. N. J. Higham: *Functions of Matrices – Theory and Computation.* SIAM (2008).
3. G. H. Golub, C. F. Van Loan: *Matrix Computations*, 4th ed. Johns Hopkins University Press (2013), Sec. 8.5.
4. O. K. Smith: Eigenvalues of a symmetric 3 × 3 matrix. *Communications of the ACM* 4(4) (1961), 168.
5. J. Kopp: Efficient numerical diagonalization of hermitian 3×3 matrices. *Int. J. Mod. Phys. C* 19 (2008), 523–548. arXiv:physics/0610206.
6. C. de Boor: Divided differences. *Surveys in Approximation Theory* 1 (2005), 46–69. arXiv:math/0502036.
7. C. Moler, C. Van Loan: Nineteen dubious ways to compute the exponential of a matrix. *SIAM Review* 20(4) (1978), 801–836. Updated version: Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later. *SIAM Review* 45(1) (2003), 3–49.
