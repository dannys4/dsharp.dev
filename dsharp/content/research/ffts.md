+++
title="Fast Fourier Transforms"
date=2021-04-13
draft=false
+++

![An illustration of the Cooley-Tukey FFT with Complex Vectorization](/posts/images/cooleytukey.svg)

The fast Fourier transform has inarguably played a role in almost all signals processing conducted today and is generally regarded among the most important and ingeneous algorithms ever developed. My research primarily focused on providing a single-dimensional implementation using vectorized complex arithmetic for the [HeFFTe Project](http://icl.utk.edu/fft/). This work was eventually published in IEEE HPEC 2021, and the paper can be found [here](https://www.icl.utk.edu/files/publications/2021/icl-utk-1497-2021.pdf). I cannot do justice to the overwhelming amount of well-written literature that explains the fast Fourier transform, let alone the Fourier transform itself, so I'd recommend looking through a few texts like [Van Loan](https://epubs.siam.org/doi/book/10.1137/1.9781611970999) or, frankly, [Wikipedia](https://en.wikipedia.org/wiki/Fast_Fourier_transform?oldformat=true#Cooley%E2%80%93Tukey_algorithm).

### For a user-friendly FFT implementation, check out my [FFTA.jl](https://github.com/dannys4/FFTA.jl) package for Julia

## CPU-vectorized Complex Arithmetic

On CPU Vectorization, this is among the most interesting topics I've delved into so far. Suppose you have $x,y\in\mathbb{C}$. You can represent these pretty easily in $\mathbb{R}^2$-- if $x = a_1 + b_1i$ and $y = c_1+d_1i$, then you get that $\mathbf{x} = (a_1,b_1)^T$ and $\mathbf{y} = (c_1,d_1)^T$. Then,

$xy = (a_1c_1-b_1d_1) + (a_1d_1+b_1c_1)i$

$\quad\\ =\begin{matrix}a_1c_1-b_1d_1 \\\\ a_1d_1+b_1c_1\end{matrix}$

$\quad\\ = \begin{bmatrix}a_1c_1\\\\a_1d_1\end{bmatrix} + \begin{bmatrix}-1 & 0\\\\0 & 1\end{bmatrix}\begin{bmatrix}b_1d_1\\\\ b_1c_1\end{bmatrix}$

$\quad\\ = \left(\begin{bmatrix}1 & 0\\\\1 & 0\end{bmatrix}\mathbf{x} \odot \mathbf{y}\right) + \begin{bmatrix}-1 & 0\\\\0 & 1\end{bmatrix}\left(\begin{bmatrix}0 & 1\\\\0 & 1\end{bmatrix}\mathbf{x} \odot \left(\begin{bmatrix}0 & 1\\\\1 & 0\end{bmatrix}\mathbf{y}\right)\right).$

We can then make $\mathbf{x},\mathbf{y}\in\mathbb{R}^{2\times N}$ have their $n$th column represent a complex number, and using the above linear algebra formula would give $\mathbf{x}_n\otimes\mathbf{y}_n$. This seems overly complicated, but it is very fast on a modern machine. Each one of these linear algebra operations can be executed in one instruction (using fused-multiply add actually gives us both $\odot$ and $+$ in one instruction). Alternatively, one can think about the transposes of $\mathbf{x}$ and $\mathbf{y}$, so we store all the real parts in the first column and all the imaginary parts in the second. Then, multiplication can be done in a fairly straight forward manner extending from the definition, i.e.

$(\mathbf{x}^T\otimes\mathbf{y}^T)_1 = \mathbf{x}^T_1\otimes\mathbf{y}^T_1 - \mathbf{x}^T_2\otimes\mathbf{y}^T_2,$

$(\mathbf{x}^T\otimes\mathbf{y}^T)_2 = \mathbf{x}_1^T\odot\mathbf{y}_2^T + \mathbf{x}_2^T\odot\mathbf{y}_1^T,$

where the subscripts each represent the column. Both of these can be very powerful when using vectorized instructions in the context of AVX, AVX2, AVX512, SVE, or other similar sets of instructions. Some information not included in the original paper linked above-- while permutation instructions used in the first expression of vectorized complex multiplication generally has a latency of about 1 clock cycle, CPUs made up until 2020 didn't have as many registers to handle these permutations, thus limiting the throughput to one per cycle. Alternatively, the fused-multiply adds used in both (and the exclusive tool employed by the second algorithm) generally have a higher latency (generally around 4 clock cycles), but the way that the CPU is structured allows one to have a throughput of about 4 per clock cycle. However, this theory is only employed in practice by the compiler _on occasion_. I've had mixed success with this, where the performance will be better between these two frameworks depending on the way this is written in terms of intrinsics, which intrinsics are used, which compiler is used, which version of the compiler is used, which CPU architecture you're on, and which model of the CPU you have. In practice, there's marginal differences, though. Another note is that, if you implement the latter version, you're better off using two "vector packs", one packing in the real parts of each number (e.g. $\mathbf{x}_1^T$) and the second packing in the imaginary parts of each number (e.g. $\mathbf{x}_2^T$). Then, if possible, use whatever equivalent your language of choice has to `std::pair` from C++ instead of just using a custom class/struct. Generally, something like `std::pair` has significantly better performance gains when compiling down due to things that a custom struct can't easily account for.