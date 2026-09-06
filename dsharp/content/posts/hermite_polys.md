+++
title = "A few special cases of Hermite polynomials: Expectations under general Gaussians"
date=2026-08-30
draft=false
+++

Anyone who knows me knows that I love the set of Hermite polynomials; they are, truly, a beautiful set of polynomials for the probabilist. They form a complete basis under a Gaussian weighting (in the $L_2$ sense), they work quite well for approximation, and they have deep ties to physics. For example, in the age of diffusion modeling, it seems like more people should know that these polynomials form the eigenfunctions of the Ornstein--Uhlenbeck process.

Something that's come up a few times for me is a few (constructive) formulas regarding the average of Hermite polynomials under a general Gaussian, and this isn't readily apparent in the abundant set of formulas on their extremely long Wikipedia page. They include a Feldheim (integral) formula that technically subsumes what I put down here, but I don't particularly like the generality of this one. Further, everything is about physicist Hermite polynomials, which I do not like, and I thought a few derivations were in order.

This is a blog post; technically, nothing here recovers anything I know from the literature, but I also acknowledge that the literature is far wider than I wish to cover. Further, I view these pedagogically rather than of particular use.

# What are the Hermite polynomials?
There are many definitions for them, but I'm going to define the (probabilist!) Hermite polynomials using their three-term recurrence definition. For a fixed $n \geq 0$, define $\mathrm{He}\_n$ at $x\in\mathbb{R}$ as

$$
\mathrm{He}\_{n+1}(x) = x\mathrm{He}\_{n}(x) + n\mathrm{He}\_{n-1}(x),\quad \mathrm{He}\_0\equiv 1,\quad \mathrm{He}\_1(x) = x.
$$

I think it's pretty clear this is a polynomial. I won't go into much detail on why this is important, as I think the Wikipedia page does a much better job than I can.

As a point of note, I am solely focused _probabilist_ Hermite polynomials $\mathrm{He}\_n$, though I'll use some identities from physicist Hermite polynomials $H_n$; these are related through the identity:
$$
H_n(x) = 2^{\frac{n}{2}} \mathrm{He}\_n (\sqrt{2} x),\quad \mathrm{He}\_n(x) = 2^{-\frac{n}{2}} H_n\left( \\frac{x}{\sqrt{2}}\right).
$$
Another point is that _all expectations_ are over a standard Gaussian measure. For reasons that I think will be apparent, it's easier to work by transforming the space of the input of the polynomial rather than work on a more general measure.

# Some useful facts:
### Zero: TTRR consequences
A few little properties falling out of the TTRR pretty directly. First, the derivative of a Hermite polynomial is a Hermite polynomial, i.e.,
$$
\\mathrm{He}\_n^\\prime(x) = n\\mathrm{He}\_{n-1}(x).
$$
We can see this immediately from induction on the three-term recurrence:
$$\\begin{aligned}
\\mathrm{He}\_{n+1}^\\prime(x) &= \\mathrm{He}\_n(x) + x\\mathrm{He}^\prime\_n(x) + n\\mathrm{He}\_{n-1}^\\prime(x)\\\\
&=\\mathrm{He}\_{n}(x) + n(x\\mathrm{He}\_{n-1}(x) - (n-1)\\mathrm{He}\_{n-2}(x))\\\\
&= (n+1)\\mathrm{He}\_{n}(x).
\\end{aligned}$$

Another odd factoid which, perhaps surprisingly, will come in handy. We get that the behavior of the Hermite polynomial on the imaginary line will be relatively nice. We can see
$$
\\mathrm{He}\_{2k}(ix)\\in\\mathbb{R},\quad i\\mathrm{He}\_{2k-1}(ix)\\in\\mathbb{R}.
$$
Suppose we have $y_k = \\mathrm{He}\_{2k}(ix)$ and $iz_k = \\mathrm{He}\_{2k-1}(ix)$ for $y_k,z_k\in\mathbb{R}$. Then,
$$
\\mathrm{He}\_{2k+1}(ix) = (ix)y\_k - n(iz\_k) = i(xy\_k - nz\_k) = iz\_{k+1},
$$
where $z\_{k+1}\\in\\mathbb{R}$. Similarly,
$$
\\mathrm{He}\_{2k+2}(ix) = (ix)iz\_{k+1} - ny\_k = -xz\_{k+1} - ny\_k = y\_{k+1},
$$
where $y\_{k+1}\\in\\mathbb{R}$.

### One: The generating function of Hermite polynomials
Many orthogonal polynomials---e.g., Laguerre, Legendre, Jacobi---have what's called a _generating function_: For a set of polynomials (or, more broadly, a function family) denoted $p_n$ for $n\geq 0$, the generating function of this family, $F:\\mathbb{R}\\times\\mathbb{R}\\to\\mathbb{R}$, is formally defined as
$$
F(x,t) = \\sum\_{n=0}^\infty p_n(x) \\frac{t^n}{n!}.
$$
The Hermite polynomials are no exception. We get:
$$F(x,t) = \\exp\\left(xt - \\frac{1}{2}t^2\\right).$$

In fact, one can use this to define the Hermite polynomials and recover the TTRR. Consider
$$
\\frac{d}{dt}F(x,t) = (x-t)F(x,t),\\quad \\frac{d^2}{dt^2} F(x,t) = -F(x,t) + (x-t)^2 F(x,t).
$$
One can repeat this a few more times to induce the following identity:
$$
\\frac{d^{n+1}}{dt^{n+1}}F(x,t) = (x-t) \\frac{d^{n}}{dt^{n}}F(x,t) - n\\frac{d^{n-1}}{dt^{n-1}} F(x,t).
$$
We know, however, that the map $f_n(t)$ defined as $t\\mapsto\\frac{d^n}{dt^n} F(x,t)$ satisfies $f_n(0) = \\mathrm{He}\_n(x)$ as defined by the Maclaurin series expansion. Therefore, the generating function also defines this three-term expansion.

### Two: The Fourier transform of the Gaussian
This is a pretty standard result that people generally see when introduced to the (continuous) Fourier transform, so I won't dwell on it. You should just know (how to look up) that
$$
\\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\int \\exp(-ixt)\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)\\ \\mathrm{d}t = \\exp(-i\\mu t)\\exp\\left(-\\frac{1}{2}\\sigma^2 t^2\\right).
$$
This subsumes the well-known fact that the Fourier transform of a Gaussian is a scaled Gaussian; take $\\mu=0$ and $\\sigma^2=1/2$ for what people usually derive in signals and systems classes.

### Three: The Feldheim summation formula
This one is somewhat annoying. The easiest way to access this information is via the [Bateman manuscript project](https://authors.library.caltech.edu/records/cnd32-h9x80), Vol ii, Chapter 10.13. Equation (37) translated accordingly for probabilist Hermite polynomials reads:
$$\\begin{aligned}
\\mathrm{He}\_n(x)\\mathrm{He}\_m(x) &= \\sum\_{r=0}^{\\min(n,m)} r! \\binom{n}{r}\\binom{m}{r} \\mathrm{He}\_{n+m-2r}(x)\\\\
&=\\sum\_{r=0}^{\\min(n,m)} \\frac{n!m!}{r!(n-r)!(m-r)!}\\mathrm{He}\_{n+m-2r}(x).
\\end{aligned}$$

They claim, however, that this "can easily be proved by means of the generating function." Maybe I'm blind, but ridiculous statements like this are something I don't miss from the era before overly-explanatory ML math papers. For what its worth, Feldheim---[to whom this identity is sometimes credited](https://doi-org.libproxy.mit.edu/10.1112/jlms/s1-13.1.22) (Eqn 1.4)---writes this identity in French, which I do not understand. He, however, cites [a contemporaneous result by S.C. Dhar](https://dspace.bcu-iasi.ro/static/web/viewer.html?file=https://dspace.bcu-iasi.ro/bitstream/handle/123456789/43666/Bulletin%20of%20the%20Calcutta%20Mathematical%20Society%2c%201934%2c%20Vol.%20%2026.pdf?sequence=2&isAllowed=y) (Bulletin of the Calcutta Mathematical Society, Volume 26, Page 59, Eqn (7)) in [a later review](https://dwc.knaw.nl/DL/publications/PU00017406.pdf) (Page 229). Dhar's proof, while in English, relies on Cauchy integration which hardly seems necessary.

What Feldheim does, however, is a brute-force combinatorial proof (which I think makes a lot of sense). I will try to reproduce it in English, more-or-less. Without loss of generality, assume $n\leq m$. First, note a few things: We know that $\\mathrm{He}\_n\\mathrm{He}\_m$ is a polynomial of degree $n+m$, and thus can be represented by the first $n+m$ Hermite polynomials. Further, we know that even-degree Hermite polynomials are even functions and odd-degree Hermite polynomials are odd functions. Thus, the parity of $n+m$ (i.e., whether it's even or odd) determines the parity of the functions, i.e., we _know_ before doing any analysis that
$$
\\mathrm{He}\_{n}(x)\\mathrm{He}\_{m}(x) = \\sum_{r=0}^{n} a\_{r}^{(m,n)}\\mathrm{He}\_{n+m-2r}(x).
$$
This ensures that the resulting polynomial has the correct parity, since we assumed $n\leq m$ and thus $n+m-2r \geq 0$ is ensured. One can, at this point, formally use Hermite triple-products to get $a\_{r}^{(m,n)}$, i.e.,
$$
a\_{r}^{(m,n)} = \mathbb{E}[\\mathrm{He}\_{n}\\mathrm{He}\_{m}\\mathrm{He}\_{n+m-2r}],
$$
which is extremely nontrivial without such Feldheim formulas beforehand.

We're going to assume $m > n$ for easy purposes. One can do everything more carefully, though. Using the Appell property $\\mathrm{He}\_{n}^{\prime}= n\\mathrm{He}\_{n-1}$, we differentiate both sides to see that
$$
n\\mathrm{He}\_{n-1}(x)\\mathrm{He}\_{m}(x) + m\\mathrm{He}\_{n}(x)\\mathrm{He}\_{m-1}(x) = \\sum_{r=0}^{n}a\_{r}^{(m,n)}(n+m-2r)\\mathrm{He}\_{n+m-2r-1}(x).
$$
We also know
$$
\\mathrm{He}\_{n-1}(x)\\mathrm{He}\_{m}(x) = \\sum_{r=0}^{n-1} a\_{r}^{(m,n-1)}\\mathrm{He}\_{n-1+m-2r}(x).
$$
Therefore, we know:
$$
\\sum_{r=0}^{n}a\_{r}^{(m,n)}(n+m-2r)\\mathrm{He}\_{n+m-2r-1}(x) = \\sum_{r=0}^{n-1} (na\_{r}^{(m,n-1)} + ma\_{r}^{(m-1,n)})\\mathrm{He}\_{n+m-2r-1}(x).
$$
For identifiability purposes, then, we must have that
$$
(n+m-2r)a\_{r}^{(m,n)} = na\_{r}^{(m,n-1)} + ma\_{r}^{(m-1,n)}.
$$
Now we simply prove that the binomial-like term follows this property:
$$
\\begin{aligned}
n a\_{r}^{(m,n-1)} + ma\_{r}^{(m-1,n)} &= n\\frac{(n-1)!m!}{(n-1-r)!(m-r)!r!} + m\\frac{n!(m-1)!}{(n-r)!(m-1-r)!}\\\\
&= (n-r) \\frac{n!m!}{(n-r)!(m-r)!r!} + (m-r) \\frac{n!m!}{(n-r)!(m-r)!}\\\\
&= (n + m - 2r) a\_{r}^{(m,n)}.
\\end{aligned}
$$
To be completely honest, I'm not super satisfied with this proof: it's obviously complete (for $m > n$), but the combinatorial coefficient seems to kind of come out thin air. I guess this is why I work in computation and not combinatorics. Like I said, Dhar gives a completely different proof that uses the Cauchy integral formula, i.e., complex analysis. This gives a little more intuition: the factorials simply come from Gamma functions, which are characteristic of contour integral formulas.

A nice consequence of this which I won't get into is some identities of Hermite triple products:
$$
\\mathbb{E}[\\mathrm{He}\_n\\mathrm{He}\_m\\mathrm{He}\_{n+m-2r}] = \\frac{n!m!}{r!(n-r)!(m-r)!,}
$$
and that Hermite triple products are "sparse" in the index space. These are particularly helpful in many polynomial approximation papers; see, e.g., stochastic Galerkin approaches to PDEs with random coefficients.

# An account of general Gaussian formulas
We are now ready to give some characterizations of expectations of probabilist Hermite polynomials under arbitrary Gaussian probability densities.

### Lemma 1: Expectation of a transformed Hermite polynomial
First, I'm going to discuss the expectation
$$
\mathbb{E}[\mathrm{He}\_n (sX + m)],\\ s,m\in\mathbb{R},
$$
which is equivalent to the expectation of $\mathrm{He}\_n$ over an arbitrary Gaussian.

$$
\\begin{aligned}
\\sum\_{k=0}^{\\infty}\\mathrm{He}\_{k}(sx+m) \\frac{t^{k}}{k!} &= \\exp\\left((sx+m)t- \\frac{1}{2}t^{2}\\right)\\\\
\\sum\_{k=0}^{\\infty}\\mathbb{E}[\\mathrm{He}\_{k}(sX+m)] \\frac{t^{k}}{k!} &= \\mathbb{E}\\left[ \\exp\\left((sX + m)t - \\frac{1}{2} t^{2}\\right)\\right]\\\\
&= \exp\\left(- \\frac{1}{2}t^{2}\right) \mathbb{E}\left[ \exp\left(i(sX+m) \\frac{t}{i}\\right)\\right]\\\\
&= \\exp\\left(- \\frac{1}{2}t^{2}\\right) \\exp(mt)\\exp\\left(\\frac{1}{2}s^{2}t^{2}\\right)\\\\
&=\\exp\\left(mt - \\frac{1 - s^2}{2}t^2\\right)
\\end{aligned}
$$
where the second-to-last step comes from the Fourier transform of the Gaussian with mean $m$ and variance $s^2$ evaluated at $t/i$. Continuing on, it seems reasonable enough to find some $x^\\prime,t^\\prime$ such that we get the righthand side of the above expressions as a generating function evaluation $F(x^\prime, t^\prime). This would allow us to move back to a summation, then allow us to hopefully decouple each term from one another.

Recall that all of these are defined on the complex plane; then, reindex using auxiliary variables $t^\prime = \\sqrt{1-s^2}$ and $x^\\prime = m / \\sqrt{1-s^2}$. We thus get
$$\\begin{aligned}
\\exp\\left(mt - \\frac{1-s^2}{2}t^2\\right) &= \\exp\\left(x^\\prime t^\\prime - \\frac{1}{2}t^\\prime\\right)\\\\
&= F(x^\\prime, t^\\prime)\\\\
&= \\sum_{k=0}^\\infty \\mathrm{He}\_n(x^\\prime) \\frac{(t^\\prime)^k}{k!}\\\\
&= \\sum_{k=0}^\\infty \\mathrm{He}\_k\\left(\\frac{m}{\sqrt{1-s^2}}\\right)(1-s^2)^{k/2}\\ \\frac{t^k}{k!}.
\\end{aligned}$$

By, e.g., extension arguments, we must then have that:
$$
\\mathbb{E}[\\mathrm{He}\_n(sX + m)] = \\mathrm{He}\_n\\left(\\frac{m}{\sqrt{1-s^2}}\\right)(1-s^2)^{n/2}.
$$
This is, in fact, always a real quantity! When $n$ is even, evaluating $\\mathrm{He}\_n$ on the imaginary line gives a real number as shown in property 0, and $(1-s^2)$ is taken to an integer power. When $n$ is odd, the polynomial $\\mathrm{He}\_n$ will give an imaginary number but $(1-s^2)^{n/2}$ will be imaginary as well, giving a real number! Unfortunately, though, we cannot evaluate these objects easily entirely without the complex numbers without just expanding the Hermite polynomial into a weighted sum of monomials, which makes it more difficult to work with.

### Lemma 2: Expectation of a product of Hermite polynomials
The hard parts are completed: we can pretty much just use the above formula as well as the Feldheim formula:
$$\\begin{aligned}
\\mathbb{E}[\\mathrm{He}\_i(sX+m)\\mathrm{He}\_j(sX+m)] &= \\sum\_{r=0}^{\min(i,j)} \\frac{i!j!}{r!(i-r)!(j-r)!}\\mathbb{E}[\\mathrm{He}\_{i+j-2r}(sX+m)]\\\\
&= \\sum\_{r=0}^{\min(i,j)} \\frac{i!j!}{r!(i-r)!(j-r)!}\\mathrm{He}\_{i+j-2r}\\left(\\frac{m}{\sqrt{1-s^2}}\\right)(1-s^2)^{\\frac{i+j-2r}{2}}.
\\end{aligned}$$
