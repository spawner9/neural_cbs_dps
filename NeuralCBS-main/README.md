# Neural Convergent Born Series
we have Born Series as follows:
$\psi = GV\psi + GS$, 
what did the modified Born Series do is to multiply a suitable preconditioner $\gamma$:
$\gamma \psi = \gamma GV \psi + \gamma GS$
, now we can derive the iterative form: $\psi_{k+1} = \psi_k - \gamma(\psi_k - G(V\psi_k + S))$
, where $G = \mathcal{F}^{-1} \tilde{g}_0 \mathcal{F}$

