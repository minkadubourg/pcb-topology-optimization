# Method

## 1. Conduction model with pads and lumped components

A single copper layer is modelled as a 2D conductor (per unit thickness). The
board $\Omega$ carries a density $\rho \in [0,1]$ per element, and its
conductivity is

$$\sigma(\rho) = \sigma_\min + (\sigma_\max - \sigma_\min)\,\rho^p ,\qquad \sigma_\max = 1,\ \sigma_\min = 10^{-6}.$$

The potential $u$ satisfies $\nabla\cdot(\sigma\nabla u) = 0$, which follows from
$\nabla\cdot J = 0$, $J = \sigma E$ and $E = -\nabla u$.

**Pads.** Every pad $p$ (connector or component lead) is an equipotential contact
with an unknown voltage $U_p$. The potential is written

$$u = u_0 + \sum_{p} U_p\,\phi_p ,$$

where $u_0$ vanishes on the pad boundaries, and $\phi_p$ is the P1 function equal
to 1 on pad $p$ and 0 at every other node.

**Components.** A two-terminal component $c$ connects pads $a_c$ and $b_c$ with a
current–voltage law $i_c = f_c(U_{a_c} - U_{b_c})$. The current balance at each
pad (Kirchhoff's current law) is added to the weak form. For a linear resistor,
$f_c(\Delta U) = \Delta U / R_c$.

**Weak form.** Find $(u_0, U)$ such that, for all $(v_0, V)$,

$$\int_\Omega \sigma(\rho)\,\nabla u\cdot\nabla v\,dx + \sum_c \frac{1}{R_c}(U_{a_c}-U_{b_c})(V_{a_c}-V_{b_c}) = \sum_p I_p V_p ,$$

with $v = v_0 + \sum_p V_p\phi_p$. Here $I_p$ is the current injected at pad $p$.
Pads with an imposed voltage (supply, ground) have their $U_p$ fixed.

The original formulation also handles nonlinear components. It does this by
introducing the component currents as unknowns, imposed weakly with Lagrange
multipliers. For linear resistors, eliminating those multipliers gives the
symmetric system above. A symmetric system matters for the adjoint, because
the adjoint can then reuse the direct solve's LU factorisation.

**Discretisation.** The bulk uses P1 elements, plus one scalar unknown per pad
and one density per element. The reduced matrix is $K = T^\top K_{\text{nodal}} T + L$:

- $T$ maps the unknowns (interior nodes, pad voltages) to nodal values;
- $L$ is the component conductance matrix.

## 2. Adjoint gradient

Let $K(\rho)\,x = f$ be the reduced system and $J(x)$ an objective. The adjoint
state solves

$$K\,\lambda = -\frac{\partial J}{\partial x},$$

and the derivative with respect to the density of element $e$ is

$$\frac{dJ}{d\rho_e} = \lambda^\top \frac{\partial K}{\partial \rho_e}\,x = \sigma'(\rho_e)\int_e \nabla u\cdot\nabla \lambda\,dx .$$

This is the Lagrangian derivation of the original study: the derivative of
$\mathcal L(\rho,u,\lambda) = J(u) + a_\rho(u,\lambda) - \ell(\lambda)$ at the
saddle point. The tests compare it with centred finite differences at the full
$10^6$ conductivity contrast. They do this for three objectives and both
interpolation laws, and the agreement is better than $10^{-4}$ relative.

**Objectives.**

| name | $J$ | $\partial J / \partial x$ |
|---|---|---|
| voltage target | $\tfrac12(U_a - U_b - U^\star)^2$ | $\pm(U_a - U_b - U^\star)$ on the two pad unknowns |
| compliance | $\sum_p I_p U_p$ = Joule losses | $f$ (self-adjoint: $\lambda = -x$) |
| field energy | $\tfrac12\int|\nabla u|^2$ | $M x$, with $M$ the unweighted stiffness |

**Constraints.**

- Copper area: $\sum_e A_e\tilde\rho_e / A \le X$.
- Optionally, the amount of intermediate material: $\sum_e A_e\tilde\rho_e(1-\tilde\rho_e)/A \le Y$. Since $\rho(1-\rho)\le 1/4$, this constraint can only be active for $Y < 1/4$.

**Filter.** Linear density filter $\tilde\rho = W\rho$ with weights
$\max(0, r - |c_e - c_j|)\,A_j$, normalised by row. The gradients are
propagated back with $W^\top$.

## 3. Optimisers

- **Projected steepest descent** (`projected_gradient_descent`): the step is the gradient normalised by its largest value, and densities are clipped to $[0,1]$. A step that lowers $J$ is accepted and the step size grows by 1.2; otherwise the step is rejected and the size is halved. It handles bounds only.
- **MMA** (Svanberg 1987, via `mmapy`): handles the constraints. The objective is scaled to about 100 at the start, and the move limit is 0.2.

## 4. Time-harmonic A–V model

For fields $\propto e^{j\omega t}$, take $E = -j\omega A - \nabla V$ and $B = \nabla\times A$.
With $J = \sigma E$, Ampère's law with displacement current gives

$$\nabla\times(\nu\nabla\times A) + (j\omega\sigma - \omega^2\varepsilon)A + (\sigma + j\omega\varepsilon)\nabla V = J_\text{free}.$$

The scalar potential follows from conservation of the total current. In the
quasi-static regime the $j\omega A$ feedback on $V$ is neglected, which gives

$$\nabla\cdot\big((\sigma + j\omega\varepsilon)\nabla V\big) = 0 .$$

The two equations are then solved one after the other:

1. $V$, with the complex electrode voltages imposed;
2. $A$, with $n\times A = 0$ on the outer boundary.

Both are complex-symmetric problems, solved with a direct solver. Two checks
validate the solution:

- the copper planes come out equipotential and $V$ is linear across the core, to $5\times10^{-8}$;
- Ampère's law $\oint B\cdot dl = \mu_0 I_\text{enc}$ holds to $2\times10^{-4}$ on loops in the core (second-order elements).

The original study wrote the scalar equation with $\varepsilon$ alone. Using
$\sigma + j\omega\varepsilon$ instead makes conductors equipotential: with
$\varepsilon$ alone, the potential also drops linearly across the copper
planes.
