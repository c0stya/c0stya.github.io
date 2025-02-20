#link("./index.html")[Home] | #link("./about.html")[About]

== Plain autoencoder

There is a function from space $X$ to space $Z$.
Objective: maximize mutual information $I(X, Z)$. So,

$
max I(X, Z) & = H(X) - H(X|Z) \
     & = max [ - H(X|Z) ] & H(X) "is a constant" \
     & = max bb(E)_(x, y ~ P(x,y)) [ log P(x|z) ] \
     & = max bb(E)_(x ~ P(x)) [log P(x|Z=f_(theta)(x) ] & " " X arrow.r Z "is deterministic"
$

Let's take another distribution $Q$ and use it to approximate original distribution $P$.

$ bb(E)_(x ~ P(x)) [log P(x|Z=f_(theta)(x) ]
		 & = bb(E)_(x ~ P(x)) [log P(x|Z=f_(theta)(x)) Q(x|Z=f_(theta)(x)) / Q(x|Z=f_(theta)(x)) ] \
		 & = bb(E)_(x ~ P(x)) [log P(x|Z=f_(theta)(x)) / Q(x|Z=f_(theta)(x)) + log Q(x|Z=f_(theta)(x)) ] \
		 & = bb(E)_(x ~ P(x)) [log P(x|Z=f_(theta)(x)) / Q(x|Z=f_(theta)(x))] + bb(E)_(x ~ P(x)) [ log Q(x|Z=f_(theta)(x)) ] \
		 & = "KL" [ P(X|Z) | Q(X|Z) ] + bb(E)_(x ~ P(x)) [ log Q(x|Z=f_(theta)(x)) ] \
		 & >= bb(E)_(x ~ P(x)) [log Q(x|Z=f_(theta)(x)) ] $

Last step is possible becase KL-divergence is non-negative and becomes equality when the KL-divergence is zero.

We can't directly optimize the sum but we can maximize the latest term. By doing so we shift the distribution $Q$ to be close to $P$.


== Variational Autoencoder

The idea is to maximize $p_theta (x)$. If it is done we could use it to sample $x in cal(X)$. There are two problems. First, we can't estimate it directly due to dimensionality. Second, it is hard to sample from this distribution. So we have add an assuption that we generate our samples from hidden distribution $p(x|z)$. Again, there is a problem with estimation. So we rely on another distribution $q(z)$ and try to "fit" it to the true one using variational methods.

_There are some philosophical consideration why should we optimize $p(x)$ instead of generating function directly. Some methods optimize the generator directly (like GANs)._

Below is the derivation using importance sampling. We want to find the parameters $theta$ of distribution $p_theta(x)$. Let's decompose it:

$
p_theta (x) & = integral_cal(Z) p_theta (x,z) dif x
    = integral_cal(Z) p_theta (x|z) p_theta (z) dif x \
    & = bb(E)_(z ~ p_theta (z) ) [p_theta (x|z)] & "definition of expectation" \
    & = bb(E)_(z ~ q_phi (z|x) ) [p_theta (x|z) (p_theta (z))/(q_phi (z|x)) ] & "iportance sampling" p_theta (z) -> q_phi (z|x)
$

Taking the $log$:

$
    log p_theta(x) & = log bb(E)_(z ~ q_phi (z|x) ) [p_theta (x|z) (p_theta (z))/(q_phi (z|x)) ] \
    	& >= bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z) (p_theta (z))/(q_phi (z|x)) ] & "Jensen's inequality" \
    	& = bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - "KL"[q_phi (z|x) || p_theta (z) ] & "definition of KL-divergence"
$

The inequality holds from Jensen's inequality $f (bb(E)[x]) >= bb(E)[f(x)]$ where f is _concave_ (for the _convex_ $f(x)$ we change the inequality to $<=$). Our final objective function:

$ cal(L)_(theta,phi)(D) = sum_(x in D) cal(L)_(theta,phi)(x_i) =  bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - "KL"[q_phi (z|x) || p_theta (z)] $

So we reformulated the problem from absolute maximization to optimizing lower bound. TODO: the tightness of the lower bound.
We can optimize it using gradient decsent method.

Taking derivative with respect to $theta$:

$
    & nabla_theta bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - "KL"[q_phi (z|x) || p_theta (z)] \
    & = nabla_theta bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - bb(E)_(z ~ q_phi (z|x) )  [ log q_phi (z|x) - log p_theta (z)] \
    & = nabla_theta bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z) + log p_theta (z)] \
    & = bb(E)_(z ~ q_phi (z|x) ) [ nabla_theta log p_theta (x|z) +  nabla_theta log p_theta (z)] \
    & approx 1/n sum_(i=1)^n [ nabla_theta log p_theta (x|z_i) + nabla_theta log p_theta (z_i)]
$

where the last sum goes over the samples from $q_(theta)$.
Taking derivative with respect to $phi$ requires reparametrization trick (see stochastick gradients):

$
    & nabla_theta bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - "KL"[q_phi (z|x) || p_theta (z)] \
    & = nabla_theta bb(E)_(z ~ q_phi (z|x) ) [- "KL"[q_phi (z|x) || p_theta (z)] \
    & = nabla_theta bb(E)_(z ~ q_phi (z|x) ) [log p_theta(z) - log q_phi (z|x)]
$

We can't directly move gradient into the expectation because the generating probability depends on $phi$. So we use reparametrization assuming we can "push randomness" and express the $q_phi (z|x)$ as deterministic function of $x, phi$ and random variable $epsilon$ from $p_epsilon$, i.e. z = g(x, phi, epsilon).

$
    & nabla_phi bb(E)_(z ~ q_phi (z|x) ) [log p_theta (x|z)] - "KL"[q_phi (z|x) || p_theta (z)] \
    = & nabla_phi bb(E)_(z ~  q_phi (z|x)) [log p_theta(z) - log q_phi (z|x)] \
    = & nabla_phi bb(E)_(epsilon ~ p_epsilon) [log p_theta(g(x,phi,epsilon)) - log q_phi (g(x,phi,epsilon|x)] \
    = & bb(E)_(epsilon ~ p_epsilon) [ nabla_phi log p_theta(g(x,phi,epsilon)) - log q_phi (g(x,phi,epsilon|x)] \
    = & 1/n sum_(i=1)^k [ nabla_phi log p_phi(g(x, phi, epsilon_k)) -  nabla_phi log q_phi (g(x,phi,epsilon_k)|x)]
$

the later sum goes over samples of $p_epsilon$.
Finally, we have everything to estimate the model. Only thing left is to specify distribution for $p_theta (z)$, $p_theta (x|z)$, $q_phi (z|x)$ and encoder and decoder parametrization (normally neural networks).
