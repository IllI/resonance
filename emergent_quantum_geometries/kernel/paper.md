

## KERNEL-BASED GALERKIN METHODS ON COMPACT
## MANIFOLDS WITHOUT BOUNDARY, WITH AN EMPHASIS ON
## SO(3)
## A DISSERTATION SUBMITTED TO THE GRADUATE DIVISION OF
## THE UNIVERSITY OF HAWAI‘I AT M
## ̄
## ANOA IN PARTIAL
## FULFILLMENT OF THE REQUIREMENTS FOR THE DEGREE OF
## DOCTOR OF PHILOSOPHY
## IN
## MATHEMATICS
## August 2021
## By
## Patrick Collins
## Dissertation Committee:
## Thomas Hangelbroek, Chairperson
## Wayne Smith
## Rufus Willett
## Evan Gawlik
## Marcelo Kobayashi

## Abstract
We develop a kernel-based Galerkin method for numerically solving ellip-
tic partial differential equations on compact Riemannian manifolds with-
out boundary that aremesh-free,coordinate-free,regularity-preserving, and
high-performing.  A fundamental challenge is to combine a theoretical solu-
tion with mesh-freequadraturein such a way that approximation power is
not lost.
We show that the approximation power of the computed (discretized)
solution can be made to be on par with the approximation power of the the-
oretical solution, provided theoversampling  exponentis sufficiently large.
We then show how to truncate a kernel onSO(3) given by a Hilbert-Schmidt
series in such a way that the approximation power of both the truncated
solution and the discretized truncated solution (again using quadrature) is
on par with the approximation power of the theoretical solution, provided
thetruncation parameteris sufficiently large.
ii

## Contents
## 1   Introduction1
## 2   Background7
2.1    Positive Definite and Conditionally Positive Definite Kernels8
2.2    Native Spaces  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .10
2.3    Sobolev Spaces I   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .13
2.4    Kernel-Based Interpolation   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .14
2.5    Calculus on Manifolds   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .18
2.6    Sobolev Spaces II  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .21
2.7    Sets of Centers; The Zeros Lemma   .  .  .  .  .  .  .  .  .  .  .  .  .  .23
2.8    Interpolation Error Estimates  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .27
3   Kernel-Based Galerkin Methods31
3.1    Kernel-Based Quadrature   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .32
3.2    The Lagrange Basis    .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .34
3.3    Kernel-Based Galerkin Methods .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .40
3.4    The Stiffness Matrix in the Lagrange Basis   .  .  .  .  .  .  .  .  .46
4   Quadratized Kernel-Based Galerkin Approximation51
4.1    The Quadratized Stiffness Matrix .  .  .  .  .  .  .  .  .  .  .  .  .  .  .52
4.2    Error Estimates .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .67
4.3    Algorithms   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .77
## 5   The Rotation Group80
5.1    Parametrizations  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .80
5.2    Harmonic Analysis  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .83
5.3    Orthonormal Basis Eigenfunctions   .  .  .  .  .  .  .  .  .  .  .  .  .  .85
5.4    Kernels and Spaces  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .87
5.5    Energy Estimates:  Two Examples   .  .  .  .  .  .  .  .  .  .  .  .  .  .89
5.6    Algorithm:  Quadrature Weights .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .91
iii

6   A TruncatedSO(3)Kernel93
6.1    The Truncated Kernel  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .95
6.2    The Truncated Lagrange Basis   .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .97
6.3    The Truncated Stiffness Matrix  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .   108
6.4    Truncated Galerkin Approximation .  .  .  .  .  .  .  .  .  .  .  .  .  .   112
6.5    The Quadratized Truncated Stiffness Matrix .  .  .  .  .  .  .  .  .   116
6.6    Quadratized Truncated Approximation .  .  .  .  .  .  .  .  .  .  .  .   118
iv

List of Figures
4.1    Plots of log(w
min
) versus log(q
## Λ
), wherew
min
is the minimum
quadrature weight for centers Λ, for samples Λ ofSO(3) of
sizes 396, 896, 1005, 1872, and 3749.   .  .  .  .  .  .  .  .  .  .  .  .  .74
v

## Chapter 1
## Introduction
Our  overarching  goal  is  to  develop  kernel-based  methods  for  numerically
solving elliptic partial differential equations (PDEs) on compact Rieman-
nian  manifolds  that  aremesh-free,coordinate-free,high-performing,  and
regularity-preserving.   Mesh-free  means  we  don’t  need  to  build  any  addi-
tional  structure  on  our  underlying  space  -  like  triangulations  or  uniform
grids - only to sample the functions that appear in the differential equation.
Coordinate-free means when we are considering a problem on a manifold,
we don’t have to work in coordinate patches, and therefore don’t have to
join together pieces of solutions.  We note that we will work in coordinate
patches to prove various results, but the methods themselves don’t require
it.  High-performing means the methods work progressively better for more
regular solutions, and can thus be suited to the regularity of the solutions
we want to approximate.  Regularity-preserving means that the regularity
of the computed solution matches the regularity of the actual solution.
A Galerkin method is one in which the equation is put in a weak form,
which  is  an  equation  on  a  Hilbert  space  with  a  bilinear  form  on  the  left
## 1

and  a linear  functional on  the right.   Putting the  equation in  weak  form
usually requires multiplying by atest  functionand performing a manipu-
lation analogous to integration by parts.  The corresponding weak form of
the problem is then solved on a finite-dimensional subspace of that Hilbert
space.
We  demonstrate  with  a  classical  example  on  the  circle.   Consider  the
second-order equation−u
## ′′
+u=fonS
## 1
, wheref∈L
## 2
## (S
## 1
).  Our set of
test  functions  in  this  context  isH
## 1
## (S
## 1
),  the  set  of  functionsv∈L
## 2
## (S
## 1
## )
for which the (distributional) derivativev
## ′
is also inL
## 2
## (S
## 1
## ).  Multiplying
byv∈H
## 1
## (S
## 1
),  integrating,  and  performing  integration  by  parts  on  the
principal part on the left yields our weak form:
## 〈u,v〉:=
## ∫
## 2π
## 0
## (
u
## ′
## (θ)v
## ′
## (θ) +u(θ)v(θ)dθ=
## ∫
## 2π
## 0
f(θ)v(θ)
## )
dθ=:λ
f
## (v).
The  weak  formulation  of  the  problem  now  reads:  findu∈H
## 1
## (S
## 1
)  such
that〈u,v〉=λ
f
(v)  for  allv∈H
## 1
## (S
## 1
).   The  finite-dimensional  subspace
VofH
## 1
## (S
## 1
) that we use could be, for example, a space of trigonometric
polynomials, or a suitable space of piecewise-defined algebraic polynomials.
AGalerkin approximationis a solution to the following modified problem:
findu∈Vsuch  that〈u,v〉=λ
f
(v)  for  allv∈V.   Given  a  basisBfor
V, the coefficients of the Galerkin approximation in that basis are usually
obtained by solving ann×nlinear system, wheren= dim(V) - for each
b∈B,〈u,b〉=λ
f
(b) yields an equation in thenunknown coefficients, and
there arensuchb.  The matrix of this linear system is called thestiffness
matrix.
We consider in this thesis kernel-based Galerkin methods.  Other promi-
nent  Galerkin  methods  includefinite-element  methods,  where  the  finite-
dimensional spaces are functions supported in subdomains of the underly-
## 2

ing space - for example, piecewise polynomials on a triangulation.  Methods
in  which  trigonometric  polynomials  or,  more  generally,  eigenfunctions  of
the  Laplace-Beltrami  operator,  are  often  referred  to  asspectral  methods.
Indeed, the truncated Galerkin approximation scheme we treat in Chapter
6 is a spectral method.
In order to implement these methods practically we must find a way to
suitably discretize the bilinear form and linear functional.  In our case we
usequadrature, in which integrals are approximated via a weighted average
of samples of the integrand, analogous to Simpson’s rule or the trapezoid
rule in Calculus.  This poses a significant challenge.  Analyzing the error for
the problem using the true stiffness matrix is fairly elementary (we provide
this error analysis in Section 3.3).  Modifying the problem to treat a nearby
stiffness matrix whose entries are obtained with quadrature is much more
difficult. We show that, on certain manifolds and with certain kernels, there
is a method that is implementable and provides error estimates that can
be made to be on par with those from the theoretical setting,  where the
bilinear form and linear functional are computed analytically.
Initially, the goal was to do this on the rotation group,SO(3).  We used
as  an  inspiration  the  work  done  by  Narcowich,  Rowe,  and  Ward  in  [22],
in which they develop a kernel-based Galerkin method on the sphere,S
## 2
## .
The main problem here for us is that the finite-dimensional spaces which
occur naturally onSO(3) are much more poorly localized than those onS
## 2
considered in [22].  Some of our techniques work in a more general setting -
on compact manifolds without boundary.  Understanding how to do this on
SO(3) yields a way to do it on manifolds in the more general class, since all
such spaces have natural kernels defined on them that generate bases with
that kind of sub-optimal decay (but that decay nonetheless).
One special feature of the problem onS
## 2
andSO(3) is that there exist
## 3

highly-performing kernels which can be given in closed form.  For example,
on the sphere,
## Φ(x,y) =‖x−y‖
## 2
log‖x−y‖
## 2
## ,
and on the rotation group,
Φ(x,y) = sin
## (
ω
## (
y
## −1
x
## )
## 2
## )
## 3/2
## ,
whereω(x) gives the rotational angle ofx.  It may be a challenge on general
manifolds to find closed-form formulas for kernels.  In Chapter 6 we address
this  problem  by  developing  a  way  to  truncate  highly-performing  kernels
that are given by a Hilbert-Schmidt series expansion.  This gives a closed-
form kernel, which we then show has the same approximation power as the
original kernel, provided sufficiently many terms are included.  Incidentally,
our Galerkin method after we truncate is, technically, a spectral method.
We  now  discuss  two  major  points  in  this  thesis.   The  first,  found  in
Chapter 4, is made in the general context, and says, roughly speaking, that
the  numerical  approximation  we  get  from  quadrature  has  approximation
power that matches the approximation power of the theoretical approxima-
tion, provided the samples used for quadrature are sufficiently denser than
the samples used for Galerkin approximation.  The second, found in Chap-
ter 6, is made in the specific context of the rotation group,SO(3), and says,
roughly speaking, that the numerical approximation we get from truncation
has approximation power that matches the approximation power of the the-
oretical approximation, provided the truncation parameter is large enough.
We also show in Chapter 6 that, provided the truncation parameter is large
enoughandthe  samples  used  for  quadrature  are  sufficiently  denser  than
the samples used for Galerkin approximation,  that the quadratized trun-
## 4

cated Galerkin approximation has approximation power that matches the
approximation  power  of  the  theoretical  approximation.   (We  have  coined
the term “quadratized” because saying “discretized stiffness matrix” didn’t
seem right - all matrices are discrete.)
In Chapter 2, we discuss kernels, smoothness spaces (underlying Hilbert
spaces),  and present results on kernel-based interpolation.  These are the
classical problems considered in approximation theory; the results go back
to [8].
In Chapter 3 we continue with some more contemporary kernel-based
approximation results.  We discuss quadrature, energy estimates, and ex-
hibit  a  class  of  kernels  -  the  polyharmonic  kernels  -  that  provide  energy
estimates on a certain class of manifolds - the two-point homogeneous man-
ifolds.  We end the chapter with an overview of the Galerkin method, in-
cluding the theoretical results one gets without quadrature.
Our first main results appear in Chapter 4, as already mentioned.  We
establish  error  estimates  between  the  weak  solution  to  our  equation,  the
Galerkin approximation, and the quadratized Galerkin approximation.
Chapter 5 applies the results of Chapter 4 toSO(3) while developing
two key examples.  One,  the rotational surface splines,  has a closed form
but only provides an algebraic energy estimate.  The other provides an ex-
ponential energy estimate but doesn’t have a closed form.  The rotational
surface splines are ready to use, whereas the one with no closed form has
no usable formula.
In Chapter 6, we show how to work with the kernel from Chapter 5 that
has no usable formula.  We do this by truncating a Hilbert-Schmidt series
for the kernel.
Our final Chapter??gives a brief discussion of how the work in this
thesis could be continued.
## 5

At the beginning we assume only that our underlying space is a compact
topological space and our kernel is either positive definite or conditionally
positive definite.  Assumptions on both the space and the kernel are added
as they are required, until we have in chapter 4 a kernel that provides an en-
ergy estimate on a compact manifold without boundary.  Finally, in chapter
6 we will be working with a specific kernel on a specific space,SO(3).
## 6

## Chapter 2
## Background
In this chapter we lay the groundwork for subsequent chapters.  In section
2.1,  we introduce positive definite and conditionally positive definite ker-
nels, and give some examples.  We then explore the native spaces for these
kinds of kernels in section 2.2.  In section 2.3 we introduce our first notion
of  a  Sobolev  space:  theH
τ
spaces.   We  then  discuss  in  section  2.4  our
interpolation scheme and establish interpolation error estimates and some
norm-minimizing properties of certain orthogonal projections onto our ap-
proximation spaces.  While this thesis is not on differential geometry per se,
it has some basis in geometry; the geometric background we need is laid out
in section 2.5.  In section 2.6 we introduce our second notion of a Sobolev
space:  theW
m
## 2
spaces,  introduce a local metric equivalence between our
manifold and its tangent space, and prove a norm equivalence betweenH
m
andW
m
## 2
whenmis a nonnegative integer.  Section 2.7 defines various key
quantities for our sets of centers and establishes the Zeros Lemma, a crucial
result.  We end the chapter with section 2.8, in which we state and prove
an interpolation error estimate.
## 7

2.1    Positive Definite and Conditionally Posi-
tive Definite Kernels
Let Ω be a compact topological space.  Apositive definite kernelon Ω is a
continuous function Φ : Ω×Ω→Rsuch that for every finite subset Ξ of Ω
and every nonzero vectorα={α
ξ
## }
ξ∈Ξ
## ∈R
## Ξ
## ,
## ∑
ξ∈Ξ
## ∑
η∈Ξ
α
ξ
α
η
## Φ(ξ,η)>0.(2.1.1)
Equivalently, Φ is positive definite if and only if for every finite subset Ξ of
Ω, thecollocation matrix
## K
## Ξ
={Φ(ξ,η)}
ξ,η∈Ξ
is a positive definite matrix.  If an inequality as in (2.1.1) holds that is not
strict, the kernel Φ is referred to as positivesemi-definite.
If Π is a finite-dimensional subspace ofC(Ω), then Φ is calledcondition-
ally positive definitewith respect to Π on Ω if for every finite Π-unisolvent
subset Ξ of Ω and every nonzero vectorα∈R
## Ξ
for which
## ∑
ξ∈Ξ
α
ξ
p(ξ) = 0
for allp∈Π, (2.1.1) holds.  Being Π-unisolvent means that the only element
pof Π for whichp(ξ) = 0 for allξ∈Ξ isp≡0.  In this case the subspace
Π is often referred to as theauxiliary space.  So, Φ is conditionally positive
definite with respect to Π on Ω if for every finite Π-unisolvent subset Ξ of Ω,
(2.1.1) holds for all vectorsα∈R
## Ξ
that annihilate Π|
## Ξ
.  The corresponding
## 8

positive definite matrix here is theaugmented collocation matrix
## K
## Ξ,Π
## =
## 
## 
## K
## Ξ
## P
## P
## T
## 0
## 
## 
## ,
whereP={p
k
## (ξ)}
ξ∈Ξ,k=1,...,Q
, withQ= dim(Π) and{p
## 1
## ,...,p
## Q
}a basis
for Π.  So, Φ is conditionally positive definite with respect to Π on Ω if for
every finite Π-unisolvent subset Ξ of Ω, the augmented collocation matrix
## K
## Ξ,Π
is a positive definite matrix.  As before,  if for allαthat annihilate
## Π|
## Ξ
an inequality like (2.1.1) holds that is not strict,  Φ is referred to as
conditionally positivesemi-definite.
Here are some examples of positive definite and conditionally positive
definite kernels in the Euclidean setting.
Example 2.1.2.(a)  TheGaussiansΦ(x,y) =e
## −a‖x−y‖
## 2
## 2
, witha >0, are
positive definite onR
d
for alld.
(b)  Ifφis continuous and inL
## 1
## (
## R
d
## )
, then Φ(x,y) =φ(x−y) is positive
definite onR
d
if the Fourier transform
## ̂
φofφis nonnegative and non-
vanishing.  This is due to Bochner’s Theorem - see [29, Section 6.2] for
details.
## (c)  Φ(x,y) =‖x−y‖
## 2
## 2
log‖x−y‖
## 2
is conditionally positive definite onR
## 2
## .
It  is,  modulo  a  multiplicative  constant,  the  fundamental  solution  to
## ∆
## 2
onR
## 2
.  (The operator ∆ here is the usual Laplacian,
## ∂
## 2
## ∂x
## 2
## 1
## +
## ∂
## 2
## ∂x
## 2
## 2
## .)
This  means  that  forf∈C
## 4
## (
## R
## 2
## )
,  we  have  the  reproductionf(x)  =
## ∫
## R
## 2
## Φ(x,y)∆
## 2
y
f(y)dyfor allx∈R
## 2
.  The auxiliary space here is Π
## 1
, the
space of bivariate polynomials of degree at most 1.  This is one of the
kernels treated by Duchon in his seminal work, [8].
(d)  If Φ is the reproducing kernel for a reproducing kernel Hilbert spaceH
## 9

of functionsf: Ω→R, then Φ is positive semi-definite, since
## ∑
ξ∈Ξ
## ∑
η∈Ξ
α
ξ
α
η
## Φ(ξ,η) =
## ∑
ξ∈Ξ
## ∑
η∈Ξ
α
ξ
α
η
〈Φ(·,ξ),Φ(·,η)〉
## H
## =
## 〈
## ∑
ξ∈Ξ
α
ξ
## Φ(·,ξ),
## ∑
η∈Ξ
α
η
## Φ(·,η)
## 〉
## H
## =
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
α
ξ
## Φ(·,ξ)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## H
## ≥0.
Furthermore, Φ is positive definite if and only if the point evaluation
functionalsδ
x
,x∈Ω, are linearly independent inH
## ∗
## .
The types of kernels we are interested in in this thesis do not live in the
Euclidean setting.  Rather, our underlying space will be a certain type of
Riemannian manifold.
## 2.2    Native Spaces
The situation in Example 2.1.2(d) can be reversed.  A continuous positive
definite kernel Φ : Ω×Ω→Rgenerates a reproducing kernel Hilbert space
## (RKHS)N
## Φ
(Ω),  for which Φ is the reproducing kernel,  called thenative
spacefor Φ. This is achieved by first settingF
## Φ
(Ω) = span{Φ(·,x) :x∈Ω}
and equippingF
## Φ
(Ω) with the inner product
## 〈
## M
## ∑
j=1
α
j
## Φ(·,x
j
## ),
## N
## ∑
k=1
β
k
## Φ(·,y
k
## )
## 〉
## Φ
## =
## M
## ∑
j=1
## N
## ∑
k=1
α
j
β
k
## Φ(x
j
## ,y
j
## ).
This  gives  a  an  inner  product  space  (not  necessarily  complete,  so  often
referred  to  as  a  “pre-Hilbert  space”)  and  the  completion  ofF
## Φ
(Ω)  with
respect  to  the  norm‖ · ‖
## Φ
is  the  native  space  for  Φ,N
## Φ
.   The  details
of this construction can be found in Schaback,  [25],  and Wendland,  [29],
## 10

which are adaptations of earlier work by Aronszajn, [1] and [2].  The main
takeaways for our purposes are that the members of the native space are,
despite being abstract elements of a closure, continuous functions, and that
the kernel Φ is indeed the reproducing kernel forN
## Φ
(Ω).  In conjunction
with Example 2.1.2(d), this shows that there is a one-to-one correspondence
between reproducing kernel Hilbert spaces and positive definite kernels.
We assume that Ω is a finite measure space, with measureμ.  We also
assume that there are real-valued continuous functions,{φ
## `
## }
## `∈N
, that form
a complete orthonormal basis forL
## 2
(Ω).  Thus, everyf∈L
## 2
(Ω) has the
expansion
f=
## ∞
## ∑
## `=0
## ̂
f
## `
φ
## `
## ,
## ̂
f
## `
## =〈f,φ
## `
## 〉
## L
## 2
## .
(For example, spherical harmonics on the sphereS
## 2
or the Wigner-D func-
tions on the rotation groupSO(3); indeed, this will be the case whenever
we have a compact Riemannian manifold.)
We consider symmetric kernels that have an expansion of the form
## Φ(x,y) =
## ∞
## ∑
## `=0
## ̂
φ(`)φ
## `
## (x)φ
## `
## (y).(2.2.1)
If the series in (2.2.1) converges uniformly, then Φ, being the uniform limit
of continuous functions, is continuous.  Conversely, if Φ is continuous then
Mercer’s Theorem guarantees that the series in (2.2.1) converges absolutely
and uniformly (see [24, p.245]).  In section 2.8, we’ll give a sufficient condi-
tion on the coefficients of Φ to guarantee continuity of Φ.
If Φ is continuous and
## ̂
φ(`)>0 for all`∈N, then Φ is positive definite,
and the native space for Φ is
## N
## Φ
## (Ω) =
## {
f∈L
## 2
## (Ω) :
## ∞
## ∑
## `=0
## ̂
φ(`)
## −1
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## <∞
## }
## ,(2.2.2)
## 11

which is a Hilbert space with inner product
## 〈f,g〉
## N
## Φ
## (Ω)
## =
## ∞
## ∑
## `=0
## ̂
φ(`)
## −1
## ̂
f
## `
## ̂g
## `
## .
We check directly that this is a RKHS with reproducing kernel Φ.  Ifg=
Φ(·,x), then a straightforward calculation using the orthonormality of the
functionsφ
## `
showŝg
## `
## =
## ̂
φ(`)φ
## `
(x).  Hence, forf∈N
## Φ
## (Ω),
〈f,Φ(·,x)〉
## N
## Φ
## (Ω)
## =
## ∞
## ∑
## `=0
## ̂
φ(`)
## −1
## ̂
f
## `
## ̂
φ(`)φ
## `
## (x) =
## ∞
## ∑
## `=0
## ̂
f
## `
φ
## `
## (x) =f(x).
(This proves it - it is a straightforward exercise to show that reproducing
kernels are unique.)
If  Φ  is  continuous  and
## ̂
φ(`)>0  for  all` > L,  withL∈N,  then  Φ
is conditionally positive definite with respect to the auxiliary space Π
## L
## =
span{φ
## `
:  0≤`≤L}.  The native space for Φ in this context is
## N
## Φ,L
## (Ω) =
## {
f∈L
## 2
## (Ω) :
## ∞
## ∑
## `=L+1
## ̂
φ(`)
## −1
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## <∞
## }
## ,
which is a semi-Hilbert space with semi-inner product
## 〈f,g〉
## N
## Φ,L
## (Ω)
## =
## ∞
## ∑
## `=L+1
## ̂
φ(`)
## −1
## ̂
f
## `
## ̂g
## `
## .
We note that in contrast to the case of a positive definite kernel, we don’t
have reproduction; indeed, ifp∈Π
## L
then〈p,Φ(·,x)〉
## N
## Φ,L
## (Ω)
= 0 for allx∈
Ω.  If, however, Ξ is a finite Π
## L
-unisolvent subset of Ω and the coefficients
## {a
ξ
## }
ξ∈Ξ
annihilate  Π
## L
## |
## Ξ
,  then  we  do  have
## 〈
f,
## ∑
ξ∈Ξ
a
ξ
## Φ(·,ξ)
## 〉
## N
## Φ,L
## (Ω)
## =
## ∑
ξ∈Ξ
a
ξ
f(ξ).
## 12

## 2.3    Sobolev Spaces I
We now insist on slightly more structure on our underlying space; namely,
that it is a compact Riemannian manifoldM.  In this setting the Laplace-
Beltrami operator onC
## ∞
(M),−∆, has countably many positive eigenval-
ues{λ
## `
## }
## `∈N
, corresponding to eigenfunctions{φ
## `
## }
## `∈N
that form a complete
orthonormal basis forL
## 2
(M).  We are ignoring multiplicity here - the eigen-
values are not necessarily distinct, though the eigenfunctions are.
We now give our first definition of the term “Sobolev space” -H
τ
## (M).
The advantage of this definition is that it doesn’t require the computation of
any covariant derivatives - members are determined solely based on the con-
vergence of a series.  (The covariant derivative will be introduced in section
2.5.)  The drawback is that this definition can’t be used to define Sobolev
spaces onsubsetsofM.  We will explore a second definition in Section 2.6 -
## W
m
## 2
(M) - which provides a natural analog on subsets; subsequently, we will
show that for nonnegative integersmthere is a norm equivalence between
the  two  seemingly  different  spacesH
m
(M)  andW
m
## 2
(M).   From  here  on,
when the underlying space is the whole manifold, we will simply writeH
m
## ,
## W
m
## 2
## ,C
## ∞
## ,L
## 2
, etc.  instead ofH
m
## (M),W
m
## 2
## (M),C
## ∞
## (M),L
## 2
(M), etc.
Forτ≥0 theSobolev spaceH
τ
is
## H
τ
## =
## {
f∈L
## 2
## :
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
τ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## ≤∞
## }
## ,(2.3.1)
which is a Hilbert space with inner product
## 〈f,g〉
## H
τ
## =
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
τ
## ̂
f
## `
## ̂g
## `
Given the similarities between the definitions (2.2.2) and (2.3.1), it is not
hard to see that the particular kernel Φ of the form (2.2.1) with coefficients
## 13

## ̂
φ(`)  that  areexactly(1 +λ
## `
## )
## −τ
is  of  interest.   In  the  case  thatτis  a
nonnegative integer, then Φ is the fundamental solution of (1−∆)
τ
## . Indeed,
there is a broader context here.  IfQis a polynomial of degreeτand Φ is
the kernel of the form (2.2.1) with coefficients
## ̂
φ(`) =Q(λ
## `
## )
## −1
, then Φ is the
fundamental solution toQ(∆).  Formally this is justified by the following.
Since ∆ is self-adjoint andMhas no boundary,
## ∫
## M
[Q(∆)f] (y)Φ(x,y)dμ(y) =
## ∞
## ∑
## `=0
## ̂
φ(`)φ
## `
## (x)
## ∫
## M
[Q(∆)f] (y)φ
## `
## (y)dμ(y)
## =
## ∞
## ∑
## `=0
## ̂
φ(`)φ
## `
## (x)
## ∫
## M
f(y) [Q(∆)φ
## `
## ] (y)dμ(y)
## =
## ∞
## ∑
## `=0
## ̂
φ(`)Q(λ
## `
## )φ
## `
## (x)
## ∫
## M
f(y)φ
## `
## (y)dμ(y)
## =
## ∞
## ∑
## `=0
## ̂
f
## `
φ
## `
## (x) =f(x).
Remark2.3.2.Switching the integral and the summation in the very first
line above is purely formal.  In practice this would have to be justified, say
by the uniform convergence of (2.2.1), for example.
We end this subsection by noting that the eigenfunctions{φ
## `
## }
## `∈N
are
also orthogonal (though not necessarily orthonormal) with respect to the
## H
τ
inner product - the proof is straightforward.
2.4    Kernel-Based Interpolation
We interpolate on a finite subset Ξ ofM.  This means that we are given a
set of valuesc={c
ξ
## }
ξ∈Ξ
## ∈R
## Ξ
, and we want to find a functions:M→R
such  thats(ξ)  =c
ξ
for  eachξ∈Ξ.   If  Φ  is  positive  definite,  then  the
## 14

approximation spacefrom which we obtain our interpolants is
## V
## Φ,Ξ
= span{Φ(·,ξ)}
ξ∈Ξ
## .
To find the unique interpolantsfrom this space, we form the collocation
matrix
## K
## Ξ
= [Φ(ξ,η)]
ξ,η∈Ξ
## ,
which is a symmetric positive definite matrix, and then find our interpolant
s=
## ∑
ξ∈Ξ
a
ξ
## Φ(·,ξ)
by solving the linear systemK
## Ξ
a=cfor the coefficientsa={a
ξ
## }
ξ∈Ξ
## .
Often the values come from a functionf∈ N
## Φ
; i.e.c={f(ξ)}
ξ∈Ξ
## =
f|
## Ξ
.  In this case we view interpolation as an operatorI
## Ξ
fromN
## Φ
toV
## Φ,Ξ
## ;
i.e.,s=I
## Ξ
f.  Indeed,I
## Ξ
fis the orthogonal projection off∈N
## Φ
(M) onto
## V
## Φ,Ξ
in the native space inner product, since for allξ∈Ξ,
〈f−I
## Ξ
f,Φ(·,ξ)〉
## N
## Φ
=f(ξ)−I
## Ξ
f(ξ) = 0.
This  means  we  have  a  Pythagorean  Theorem:‖f‖
## 2
## N
## Φ
=‖f−I
## Ξ
f‖
## 2
## N
## Φ
## +
## ‖I
## Ξ
f‖
## 2
## N
## Φ
; consequently, we have the following.
Proposition 2.4.1.(Native Space Interpolation Error Estimate for Posi-
tive Definite Kernels)SupposeΦis positive definite andΞis a finite subset
ofM.  Letf∈N
## Φ
(M)and letI
## Ξ
fbe the interpolant tofinV
## Ξ
## .  Then
‖f−I
## Ξ
f‖
## N
## Φ
## (M)
## ≤‖f‖
## N
## Φ
## (M)
## .
Now  suppose  Φ  is  conditionally  positive  definite  with  respect  to  Π
## L
## ,
## 15

and that Ξ is a finite,  Π
## L
-unisolvent subset ofM.  The matrixK
## Ξ
is no
longer  guaranteed  to  be  positive  definite,  or  indeed  even  invertible.   Let
Q= dim(Π
## L
) and{p
## 1
## ,...,p
## Q
}be a basis for Π
## L
, and construct the matrix
## P= [p
k
## (ξ)]
ξ∈Ξ,k∈{1,...,Q}
.  Then the augmented collocation matrix
## K
## Ξ,L
## =
## 
## 
## K
## Ξ
## P
## P
## T
## 0
## 
## 
## ,(2.4.2)
ispositive definite.  For a conditionally positive definite kernel, the inter-
polant is
s=
## ∑
ξ∈Ξ
a
ξ
## Φ(·,ξ) +
## Q
## ∑
k=1
b
k
p
k
## ,
where the coefficientsa={a
ξ
## }
ξ∈Ξ
andb={b
k
## }
k∈{1,...,Q}
are obtained by
solving the linear system
## K
## Ξ,L
## 
## 
a
b
## 
## 
## =
## 
## 
c
## 0
## 
## 
## .
Theapproximation spacein the conditionally positive definite setting is
## V
## Φ,Ξ,L
## =
## 
## 
## 
## ∑
ξ∈Ξ
a
ξ
## Φ(·,ξ) :
## ∑
ξ∈Ξ
a
ξ
p(ξ) = 0∀p∈Π
## L
## 
## 
## 
## + Π
## L
## .
We say in this context that for
## ∑
ξ∈Ξ
a
ξ
Φ(·,ξ) +p∈V
## Φ,Ξ,L
, the coefficients
a={a
ξ
## }
ξ∈Ξ
annihilateΠ
## L
## |
## Ξ
## .
As before, if the values come from a functionf∈ N
## Φ,L
, then we view
interpolation as an operatorI
## Ξ,L
fromN
## Φ,L
toV
## Ξ,L
; i.e.,s=I
## Ξ,L
f.  We
note that in this situation interpolation reproduces the auxiliary space Π
## L
## ;
i.e.,I
## Ξ,L
p=pfor allp∈Π
## L
. It is also important to note that just as before,
## I
## Ξ,L
fis  the  orthogonal  projection  off∈ N
## Φ,L
(M)  toV
## Ξ,L
in  the  semi-
## 16

inner product〈·,·〉
## N
## Φ,L
## (M)
.  This again means that we have a “Pythagorean
## Theorem”‖f‖
## 2
## N
## Φ,L
=‖f−I
## Ξ,L
f‖
## 2
## N
## Φ,L
## +‖I
## Ξ,L
f‖
## 2
## N
## Φ
, and again we have the
following consequence.
Proposition 2.4.3.(Native Space Interpolation Error Estimate for Con-
ditionally Positive Definite Kernels)SupposeΦis conditionally positive def-
inite  with  respect  toΠ
## L
andΞis  a  finiteΠ
## L
-unisolvent  subset  ofM.  Let
f∈N
## Φ,L
(M)and letI
## Ξ,L
fbe the interpolant tofinV
## Ξ,L
## .  Then
‖f−I
## Ξ,L
f‖
## N
## Φ,L
## (M)
## ≤‖f‖
## N
## Φ,L
## (M)
## .
The fact that the interpolation operator is the orthogonal projector of
## N
## Φ
ontoV
## Ξ
when  Φ  is  positive  definite  has  another  consequence.   Since
## I
## Ξ
fis  theuniqueelementuofV
## Φ,Ξ
such  thatu|
## Ξ
## =f|
## Ξ
,  ifgis  any
other  element  ofN
## Φ
for  whichg|
## Ξ
## =f|
## Ξ
,  thenI
## Ξ
g=I
## Ξ
f.   Thus,  the
”Pythagorean Theorem”‖g‖
## 2
## N
## Φ
=‖g−I
## Ξ
g‖
## 2
## N
## Φ
## +‖I
## Ξ
g‖
## 2
## N
## Φ
also gives that
## ‖I
## Ξ
f‖
## N
## Φ
## =‖I
## Ξ
g‖
## N
## Φ
## ≤‖g‖
## N
## Φ
## .
We summarize in the following result.
Proposition  2.4.4.IfΦis  positive  definite  andf∈ N
## Φ
,  then  for  ev-
ery  finite  subsetΞofM,I
## Ξ
fis  the  unique  native  space  norm-minimizing
memberuofN
## Φ
such thatu|
## Ξ
## =f|
## Ξ
## .
Similar considerations gives an analogous result forI
## Ξ,L
## .
Proposition 2.4.5.IfΦis  conditionally  positive  definite  with  respect  to
## Π
## L
andf∈ N
## Φ,L
,  then  for  every  finiteΠ
## L
-unisolvent  subset  ofM,I
## Ξ,L
f
is  the  unique  native  space  semi-norm-minimizing  memberuofN
## Φ,L
such
thatu|
## Ξ
## =f|
## Ξ
## .
## 17

2.5    Calculus on Manifolds
The geometric background we need is developed in [6] and [7], and detailed
in [15] and [16].  We give a brief summary here.  We assumeMis a compact
## C
## ∞
Riemannian  manifold  without  boundary.   Letgbe  the  Riemannian
metric ofM.
The  metricgdefines  an  inner  product〈·,·〉
g,p
on  each  tangent  space
## T
p
M. We will usually omit the metricgand simply write〈·,·〉
p
to denote the
metric atp∈M.  An orderkcovariant tensoris a real-valued multilinear
function of thek-fold product ofT
p
M.  Thus, the set of orderkcovariant
tensors  is  thek-fold  tensor  product
## (
## T
## ∗
p
## M
## )
## ⊗k
## =T
## ∗
p
## M⊗ ··· ⊗T
## ∗
p
## M.   In
coordinates, there is a smoothly varying basis
## {
e
i
## 1
## ⊗···⊗e
i
k
## }
## ˆı∈{1,...,d}
k
## ,
where we adopt the convention ˆı= (i
## 1
## ,...,i
k
).  So, a covariant tensorT
can be written as
## T=
## ∑
## ˆı∈{1,...,d}
k
## T
## ˆı
e
i
## 1
## ⊗···⊗e
i
k
## ,
and theT
## ˆı
are called thecovariant componentsofT.  The metricgis itself
an order 2 covariant tensor.  Similarly, an orderkcontravariant tensor is
an element of thek-fold tensor product (T
p
## M)
## ⊗k
## =T
p
## M⊗...⊗T
p
## M.  One
can also define tensors of mixed type.
Thecovariant derivative, orconnection,∇, is defined as follows.  IfTis
an orderkcovariant tensor as above, then
## ∇T=
## ∑
## ˆı∈{1,...,d}
k
d
## ∑
j=1
## (∇T)
## ˆı,j
e
i
## 1
## ⊗···⊗e
i
k
## ⊗e
j
## ,
## 18

where the covariant components of∇Tare given by
## (∇T)
## ˆı,j
## =
## ∂T
## ˆı
## ∂x
j
## −
k
## ∑
r=1
d
## ∑
s=1
## Γ
s
j,i
r
## T
i
## 1
## ,...,i
r−1
## ,s,i
r+1
## ,...,i
k
## .
Here,  the Γ
k
i,j
’s are Christoffel symbols.  Thus,  ifTis an orderktensor,
∇Tis an orderk+ 1 tensor.
The metricginduces an invariant inner product on
## (
## T
## ∗
p
## M
## )
## ⊗k
## :
## 〈T,S〉
p
## =
## ∑
## ˆı,ˆ∈{1,...,d}
k
g
i
## 1
j
## 1
## ···g
i
k
j
k
## S
## ˆı
## T
## ˆ
## .(2.5.1)
## Theadjoint∇
## ∗
of∇is defined by
## ∫
## M
## 〈∇T,S〉
p
dμ(p) =
## ∫
## M
## 〈T,∇
## ∗
## S〉
p
dμ(p),
wheneverTis a rankktensor andSis a rankk+ 1 tensor.  So, it takes
orderk+ 1 tensors to orderktensors.
A  smooth  functionf:M−→Ris  an  order  0  tensor;  whereas∇f
is  an  order  1  (covariant)  tensor.   In  general,∇
k
fis  an  orderktensor.
In  coordinates  (U,φ),  whereφ:U→U⊂R
d
is  a  chart  andφ(p)  =
## (
x
## 1
## ,...,x
d
## )
∈U, the components of∇
k
fare
## (
## ∇
k
f(x)
## )
## ˆı
## =
## (
## ∂
k
f(x)
## )
## ˆı
## +
k−1
## ∑
m=1
## ∑
## ˆ∈{1,...,d}
m
## A
## ˆ
## ˆı
## (x) (∂
m
f(x))
## ˆ
## ,
where
## (∂
m
f(x))
## ˆ
## =
## ∂
m
## ∂x
j
## 1
## ·∂x
j
m
f◦φ
## −1
## .
Here  the  coefficientsA
## ˆ
## ˆı
(x)  depend  on  the  Christoffel  symbols  and  their
derivatives to orderk−1, and hence are smooth inx.
## Applying∇
## ∗
to  the  orderktensor∇
k
f ktimes  produces  a  scalar,
## 19

## (∇
## ∗
## )
k
## ∇
k
f,  which  is  the  same  as
## (
## ∇
k
## )
## ∗
## ∇
k
f.   Whenk=  1,  we  have  the
Laplace-Beltramioperator,−∆ =∇
## ∗
∇.  In coordinates,
## ∆f=
## 1
## √
det(g)
d
## ∑
j=1
d
## ∑
k=1
## ∂
## ∂x
j
## [
g
jk
## √
det(g)
## ∂f
## ∂x
k
## ]
## .
Crucially,  it  isnotnecessarily  the  case  that  fork >1,  ∆
k
is  equal  to
## (−1)
k
## (
## ∇
k
## )
## ∗
## ∇
k
, though it is the case if the manifold is flat (like Euclidean
space  or  the  flat  torus).   This  is  because∇and∇
## ∗
do  not  necessarily
commute in general.  For example, ∆
## 2
## =∇
## ∗
## ∇∇
## ∗
∇, whereas
## (
## ∇
## 2
## )
## ∗
## ∇
## 2
## =
## (∇
## ∗
## )
## 2
## ∇
## 2
## =∇
## ∗
## ∇
## ∗
∇∇, and these are not necessarily the same.
We do, however, have the following propositions, which will allow us to
prove a norm equivalence between our different notions of Sobolev spaces
in the next section.  The first is due to Helgason - [18] - but is stated here
as it appears in [16, Lemma 4.2 and Assumption 4.1].  We state both and
prove the second in the special case thatMistwo-point homogeneous, but it
holds for all compact manifolds without boundary.  Two point homogeneity
means that for every two pairs of pointsp,qandp
## ′
## ,q
## ′
inMwith dist(p,q) =
dist(p
## ′
## ,q
## ′
), there is an isometryψ:M→Mwithψ(p) =p
## ′
andψ(q) =q
## ′
## .
Intuitively, this means that the manifoldM“looks the same” at every point.
Proposition 2.5.2.LetMbe  a  two-point  homogeneous  space.   Then  for
allk∈Nthere is a real polynomialp
k−1
of degreek−1such that
## (
## ∇
k
## )
## ∗
## ∇
k
## = (−1)
k
## ∆
k
## +p
k−1
## (∆).
Proposition 2.5.3.LetMbe a two-point homogeneous space.  IfQ(x) =
c
m
x
m
## +...+c
## 0
is  a  real  polynomial  of  degreem,  then  there  exist  real
## 20

numbersa
j
,j∈{0,...,m}such that
## Q(∆) =
m
## ∑
j=0
a
j
## (
## ∇
j
## )
## ∗
## ∇
j
## .
Proof.Clearly it suffices to show the proposition is true whenQ(x) =x
k
fork∈N.  We proceed by strong induction.  The base casem= 1 is trivial,
since ∆ =−∇
## ∗
∇.  Suppose the proposition is true fork∈ {1,...,m}.  By
## Proposition 2.5.2,
## (
## ∇
m+1
## )
## ∗
## ∇
m+1
## = (−1)
m+1
## ∆
m+1
## +p
m
## (∆)
for a polynomialp
m
of degreem.  Letp
m
## (x) =
## ∑
m
k=0
b
k
x
k
.  By our strong
inductive hypothesis, for eachk∈ {0,...,m}there are numbersc
k,j
such
that ∆
k
## =
## ∑
k
j=0
c
k,j
## (
## ∇
k
## )
## ∗
## ∇
k
## .  Hence,
## ∆
m+1
## = (−1)
m+1
## (
## ∇
m+1
## )
## ∗
## ∇
m+1
## −(−1)
m+1
m
## ∑
k=0
b
k
k
## ∑
j=0
c
k,j
## (
## ∇
j
## )
## ∗
## ∇
j
## = (−1)
m+1
## (
## ∇
m+1
## )
## ∗
## ∇
m+1
## +
m
## ∑
j=0
## 
## 
## (−1)
m
m
## ∑
k=j
b
k
c
k,j
## 
## 
## (
## ∇
j
## )
## ∗
## ∇
j
## .
Hence we can takea
m+1
## = (−1)
m+1
anda
j
## = (−1)
m
## ∑
m
k=j
b
k
c
k,j
forj∈
## {0,...,m}.
2.6    Sobolev Spaces II
Now that we have the covariant derivative, we are ready to give our second
definition of the term “Sobolev space”.  For an integermand a measurable
## 21

subset Ω ofM, theSobolev spaceW
m
## 2
(Ω) is
## W
m
## 2
## (Ω) =
## {
f∈L
## 2
## (Ω) :
m
## ∑
k=0
## ∫
## Ω
## 〈
## ∇
k
f,∇
k
f
## 〉
p
dμ(p)<∞
## }
## ,
where the inner product in the integrand is given in (2.5.1).  It is a Hilbert
space with inner product
## 〈f,g〉
## W
m
## 2
## (Ω)
## =
m
## ∑
k=0
## ∫
## Ω
## 〈
## ∇
k
f,∇
k
g
## 〉
p
dμ(p).
Again, when Ω is all ofM, we denote it simply byW
m
## 2
instead ofW
m
## 2
## (M).
An  important  tool  for  obtaining  interpolation  error  estimates  is  the
Zeros Lemma, which will appear in the next section.  We’d like to have a
similar result on our Sobolev spaces, and [15, Lemma 3.2] allows us to do
so.
Before we state [15, Lemma 3.2], we need to define the term “injectivity
radius”.  At a pointpon a Riemannian manifoldM, theexponential  map
atp, Exp
p
## :T
x
M→M, is defined as follows.  For eachv∈T
p
Mthere is a
unique geodesicγ
v
inMsuch thatγ(0) =pandγ
## ′
(0) =v. Define exp
p
## (v) =
γ
v
(1).   The  exponential  map  is  in  general  not  a  global  diffeomorphism;
however,  at  the  origin  of  the  tangent  space,  the  differential  of  Exp
p
is
the identity.  So, the Inverse Function Theorem guarantees that there is a
neighborhood of the origin on which Exp
p
is injective.  The supremum of all
radiirfor which Exp
p
is injective onB(0,r)⊂T
p
(M) is called theinjectivity
radiusofMatp, denotedr
p
.  Theinjectivity  radiusofM, denotedr
## M
, is
the infimum ofr
p
aspranges over all ofM.  If 0< r
## M
≤∞, thenMis said
to have positive injectivity radius.  This is the case for compact manifolds.
Lemma 2.6.1.([15, Lemma 3.2])Letm∈Nand0< r < r
## M
## /3.  There
are constants0< c
## 1
## ≤c
## 2
so that for any measurableΩ⊂B(0,r), allj∈N
## 22

withj≤m, and anyp∈M, the equivalence
c
## 1
## ∥
## ∥
u◦Exp
p
## ∥
## ∥
## W
j
## 2
## (Ω)
## ≤‖u‖
## W
j
## 2
## (
## Exp
p
## (Ω)
## )
## ≤c
## 2
## ∥
## ∥
u◦Exp
p
## ∥
## ∥
## W
j
## 2
## (Ω)
holds for allu: Exp
x
(Ω)→RinW
m
## 2
## (
## Exp
p
## (Ω)
## )
.  The constantsc
## 1
andc
## 2
depend onrandmbut are independent ofΩandp.
For an open subset Ω ofR
d
that is bounded and has Lipschitz boundary,
the  Sobolev  embedding  theorem  guarantees  thatW
m
## 2
(Ω)⊂C(Ω)  for  a
subset  Ω  ofR
d
whenm > d/2.   Lemma  2.6.1  says  that  this  holds  on
manifolds  as  well;  ifm > d/2  and  Ω⊂Mis  open,  bounded,  and  has
Lipschitz boundary, thenW
m
## 2
## (Ω)⊂C(Ω).
2.7    Sets of Centers; The Zeros Lemma
At this point we need to introduce three quantities that measure the so-
called “uniformity” of our set of centers Ξ (sometimes called nodes).  The
first is thefill distance(sometimes called themesh norm),
h
## Ξ
= max
p∈M
dist(p,Ξ) = max
p∈M
min
ξ∈Ξ
dist(p,ξ).
It measures the density of the centers inM.  The second is theseparation
distance,
q
## Ξ
## =
## 1
## 2
min
ξ∈Ξ
dist(ξ,Ξ\ξ) =
## 1
## 2
min
ξ∈Ξ
min
η∈Ξ
η6=ξ
dist(ξ,η).
It measures how evenly distributed the centers are.  The third is themesh
ratio,ρ
## Ξ
## =h
## Ξ
## /q
## Ξ
.  Note that the mesh ratio is at least 2, since the defini-
tions forceh
## Ξ
## ≥2q
## Ξ
.  Sets of centers for which the mesh ratio is bounded
by some fixedρ(preferably close to 2) are sometimes calledquasi-uniform.
We will assume throughout the remainder of this paper that our sets of cen-
## 23

ters are quasi-uniform with mesh ratios controlled above by a fixedρ≥2.
In the sequel we will allow the constants in our estimates to depend onρ.
We note that in the context of a finite, quasi-uniform set of centers Ξ in
a compactd-dimensional manifoldM, there are constantsC
## 1
andC
## 2
such
thatC
## 1
h
## −d
## Ξ
## ≤#Ξ≤C
## 2
q
## −d
## Ξ
.  We also note thath
## Ξ
## =ρq
## Ξ
, and therefore,
since we allow the constants in our estimates to depend onρ,h
## Ξ
andq
## Ξ
are interchangeable.
We now state the Zeros Lemma, which, roughly speaking, says that a
weak norm of a function with many zeros can be controlled by a stronger
norm with a multiplicative constant that decreases with the density of the
zeros.  A zeros lemma was first stated and proved in [8], where the under-
lying space was a Euclidean ball.  Narcowich, Ward, and Wendland extend
the result to a broader class of spaces in [23].  In [16, Appendix A], Hangel-
broek, Narcowich, and Ward extend it to manifolds, which is the version
we use.
Lemma 2.7.1.(The Zeros Lemma for Manifolds - [16,  Corollary A.13])
Letm,k∈Nsatisfym > d/2,0≤k≤m.  In  addition,  letΞbe  a  finite
subset ofM.  Ifu∈W
m
## 2
satisfiesu|
## Ξ
= 0, then forh
## Ξ
sufficiently small,
## ‖u‖
## W
k
## 2
≤Ch
m−k
## Ξ
## ‖u‖
## W
m
## 2
## ,
and
## ‖u‖
## L
## ∞
≤Ch
m−
d
## 2
## Ξ
## ‖u‖
## W
m
## 2
## .
with a constantCthat is independent ofΞandu.
We now show the norm equivalence betweenH
m
andW
m
## 2
whenmis a
nonnegative integer.  Two important points to note are that
## 〈f,g〉
## H
m
## =〈(I−∆)
m
f,f〉
## L
## 2
## ,
## 24

whereas
## 〈f,g〉
## W
m
## 2
## =
m
## ∑
k=0
## 〈
## ∇
k
f,∇
k
f
## 〉
## L
## 2
## .
Remark2.7.2.We include the following result, and its proof, in the interest
of  self-containment.   It  is  a  simplified  version  of  a  much  broader  result
found in [28].  Let us add a bit of historical context.  Aubin introduced the
## W
m
## 2
Sobolev  spaces  on  Riemannian  manifolds  in  [3],  whereas  Strichartz
introduced  theH
m
Sobolev  spaces  on  Riemannian  manifolds  in  [27].   In
[28], Triebel showed these spaces coincide for much larger and more general
classes of function spaces than those that we will be considering.  The full
result found in [28] is, of course, true, but requires a far greater amount of
analysis than is necessary for our context.
Proposition 2.7.3.[28, Theorem 4.1]SupposeMis  a  compact  Rieman-
nian  manifold  without  boundary.  For  a  nonnegative  integerm,  the  spaces
## H
m
(M)andW
m
## 2
(M)are the same, and the norms‖·‖
## H
m
and‖·‖
## W
m
## 2
are
equivalent.
Proof in the case thatMis two-point homogeneous.Proposition 2.5.3 with
## Q(x) = (1−x)
m
gives real numbersa
j
,j= 0,...,mfor which (1−∆)
m
## =
## ∑
m
j=0
a
j
## (
## ∇
j
## )
## ∗
## ∇
j
.  This gives
## ‖f‖
## 2
## H
m
## =
m
## ∑
j=0
a
j
## 〈
## (
## ∇
j
## )
## ∗
## ∇
j
f,f
## 〉
## L
## 2
## ≤C
m
## ∑
j=0
## 〈
## ∇
j
f,∇
j
f
## 〉
## L
## 2
=C‖f‖
## W
m
## 2
## ,
where we have takenC= max
j∈{0,...,m}
## |a
j
## |.
For the other direction, we start by using Theorem 2.5.2 to obtain
## ‖f‖
## 2
## W
m
## 2
## =
m
## ∑
k=0
## 〈(
## (−1)
k
## ∆
k
## +p
k−1
## (∆)
## )
f,f
## 〉
## L
## 2
## .(2.7.4)
## 25

## Now,
## ∑
m
k=0
## (
## (−1)
k
x
k
## +p
k−1
## (x)
## )
is a polynomial of degreem.  Write it as
## ∑
m
j=0
b
j
x
j
to obtain
## ‖f‖
## W
m
## 2
## =
m
## ∑
j=0
b
j
## 〈
## ∆
j
f,f
## 〉
## L
## 2
## ≤
m
## ∑
j=0
## |b
j
## |
## ∣
## ∣
## ∣
## 〈
## ∆
j
f,f
## 〉
## L
## 2
## ∣
## ∣
## ∣
## .
Note  that  since−∆  is  positive  definite,
## ∣
## ∣
## ∣
## 〈
## ∆
j
f,f
## 〉
## L
## 2
## ∣
## ∣
## ∣
## =
## 〈
## (−1)
j
## ∆
j
f,f
## 〉
## L
## 2
for eachj.  LettingC= max
j∈{0,...,m}
## |b
j
|, we arrive at
## ‖f‖
## W
m
## 2
## ≤C
m
## ∑
j=0
## 〈
## (−1)
j
## ∆
j
f,f
## 〉
## L
## 2
## .
## But
## 〈
## (−1)
j
## ∆
j
f,f
## 〉
## L
## 2
## ≤
## 〈
## (I−∆)
j
f,f
## 〉
## L
## 2
, since the term on the left is but
one  of  the  positive  terms  that  appears  in  the  binomial  expansion  of  the
term on the right.  Hence,
## ‖f‖
## W
m
## 2
## ≤C
m
## ∑
j=0
## 〈
## (I−∆)
j
f,f
## 〉
## L
## 2
## =C
m
## ∑
j=0
## ‖f‖
## H
j
≤C(m+ 1)‖f‖
## H
m
## ,
where in the last inequality we have used the fact that‖·‖
## H
j
## ≤‖·‖
## H
m
for
## 0≤j≤m.
We  end  this  subsection  with  a  Lemma  from  [5]  that  deals  with  the
Sobolev  norms  of  products  of  functions  in  Sobolev  spaces  which  will  be
useful in the sequel.  The version of this which is used here is from Couldon,
et al.  - it extends to manifolds an earlier result (the “generalized Leibniz
rule”) onR
d
by Gulisashvili and Kon, [13, Theorem 1.4].
Lemma 2.7.5.([5, Theorem 27])Letf,gbe  inH
m
## ∩L
## ∞
,  wherem∈N
satisfiesm > d/2.  Thenfg∈H
m
## ∩L
## ∞
and there exists aC >0such that
## ‖fg‖
## H
m
## ≤C
## (
## ‖f‖
## H
m
## ‖g‖
## L
## ∞
## +‖f‖
## L
## ∞
## ‖g‖
## H
m
## )
## .
## 26

## 2.8    Interpolation Error Estimates
We proceed by adding a further assumption, this time not on the manifold
M, but on the coefficients of the kernel Φ.  Specifically, we assume that Φ
is a kernel of the form (2.2.1) whose coefficients satisfy
## C
## 1
## (1 +λ
## `
## )
## −τ
## ≤
## ̂
φ(`)≤C
## 2
## (1 +λ
## `
## )
## −τ
## .(2.8.1)
forτ > d/2, either for all`∈Nif Φ is positive definite, or for all` > L∈N
if Φ is conditionally positive definite with respect to Π
## L
## .
In the case where Φ is positive definite, it is clear from the definitions
(2.3.1) and (2.2.2) that the native space norm‖·‖
## N
## Φ
and the Sobolev norm
## ‖·‖
## H
τ
are equivalent.  This, along with the Zeros Lemma, is all we need
to prove our interpolation error estimate in the case of a positive definite
kernel.  We note first that ifτ=m∈Nwithm > d/2, then
## N
## Φ
## =H
m
## =W
m
## 2
## ⊂C(M).
(The last equality is due to the Sobolev embedding theorem.)
Theorem 2.8.2.SupposeΦis a positive definite kernel of the form (2.2.1)
whose coefficients satisfy (2.8.1) for someτ∈Nwithτ > d/2and all`∈N,
and  letσ∈Nwith0≤σ≤τ.   LetΞbe  a  finite  subset  ofM,f∈H
τ
## ,
andI
## Ξ
fthe  interpolant  tofinV
## Φ,Ξ
.  There  exists  a  constantCwhich  is
independent offandΞsuch that, forh
## Ξ
sufficiently small,
‖f−I
## Ξ
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## ‖f‖
## H
τ
## .
## 27

Proof.By the Zeros Lemma,
‖f−I
## Ξ
f‖
## H
σ
≤Ch
τ−σ
## Ξ
‖f−I
## Ξ
f‖
## H
τ
## .
Since the coefficients of Φ satisfy (2.8.1), the norms‖·‖
## H
τ
and‖·‖
## N
## Φ
are
equivalent, and so
‖f−I
## Ξ
f‖
## H
σ
≤Ch
τ−σ
## Ξ
‖f−I
## Ξ
f‖
## N
## Φ
## .
By Proposition 2.4.1, then,
‖f−I
## Ξ
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## ‖f‖
## N
## Φ
## .
Finally, using again the equivalence of those norms, we have
‖f−I
## Ξ
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## ‖f‖
## H
τ
## .
For a kernel that is conditionally positive definite with respect to Π
## L
## ,
and whose coefficients satisfy (2.8.1) for someτand all` > L, the proof
of  the  error  estimate  is  a  bit  more  involved,  though  the  result  itself  is
the  same.   The  reason  for  this  is  that  the  native  space  semi-norm  isn’t
equivalent  to  the  Sobolev  norm.   There  is  a  remedy,  however,  involving
orthogonal projections.
We require the orthogonal projectionP
## L
which projects all the Sobolev
spaces onto Π
## L
.  The projection is the following:  iff=
## ∑
## ∞
## `=0
## ̂
f
## `
φ
## `
## ∈L
## 2
## ,
thenP
## L
f=
## ∑
## L
## `=0
## ̂
f
## `
φ
## `
.  This operator is the orthogonal projection onto
## Π
## L
, because iff∈L
## 2
and`∈{0,...,L}, then
## ̂
## (P
## L
f)
## `
## =
## ̂
f
## `
, and ifg∈Π
## L
## ,
## 28

then̂g
## `
= 0 for` > L.  Therefore iff∈H
σ
for anyσ≥0,
〈f−P
## L
f,g〉
## H
σ
## =
## L
## ∑
## `=0
## (1 +λ
## `
## )
σ
## (
## ̂
f
## `
## −
## ̂
## (P
## L
f)
## `
## )
## ̂g
## `
## = 0.
Our interpolation error estimate will rely on two facts.  First, forf∈H
τ
## ,
we have by (2.8.1) that
## ‖(I−P
## L
## )f‖
## 2
## H
τ
## =
## ∞
## ∑
## `=L+1
## (1+λ
## `
## )
τ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## ≤C
## −1
## 1
## ∞
## ∑
## `=L+1
## ̂
φ(`)
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
=C‖f‖
## 2
## N
## Φ,L
## .
## (2.8.3)
Second, if 0≤σ≤τandf∈H
τ
, then
## ‖P
## L
f‖
## 2
## H
τ
## =
## L
## ∑
## `=0
## (1+λ
## `
## )
τ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## ≤C
σ,τ,L
## L
## ∑
## `=0
## (1+λ
## `
## )
σ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## =C
σ,τ,L
## ‖P
## L
f‖
## 2
## H
σ
## ,
## (2.8.4)
whereC
σ,τ,L
## = (1 +λ
## L
## )
τ−σ
depends only onσ,τ, andL.
Theorem  2.8.5.SupposeΦis  a  kernel  of  the  form  (2.2.1)  whose  coef-
ficients  satisfy  (2.8.1)  forτ∈Nwithτ > d/2and  all` > L∈N,  and
is  conditionally  positive  definite  with  respect  toΠ
## L
,  and  letσ∈Nwith
0≤σ≤τ.  LetΞbe a finiteΠ
## L
-unisolvent subset ofM,f∈H
τ
, andI
## Ξ,L
f
the interpolant tofinV
## Ξ,L
.  There exists a constantCwhich is independent
offandΞsuch that, forh
## Ξ
sufficiently small,
‖f−I
## Ξ,L
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## ‖f‖
## H
τ
## .
Proof.The caseσ=τis just Proposition 2.4.3, so assumeσ < τ.  By the
## Zeros Lemma,
‖f−I
## Ξ,L
f‖
## H
σ
≤Ch
τ−σ
## Ξ
‖f−I
## Ξ,L
f‖
## H
τ
## .
## 29

By “smuggling in” bothP
## L
fandP
## L
## I
## Ξ,L
f, we obtain
‖f−I
## Ξ,L
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## (
## ‖P
## L
(f−I
## Ξ,L
f)‖
## H
τ
## +‖(I−P
## L
) (f−I
## Ξ,L
f)‖
## H
τ
## )
## .
## Now,
## ‖P
## L
(f−I
## Ξ,L
f)‖
## H
τ
## ≤C‖P
## L
(f−I
## Ξ,L
f)‖
## H
σ
by (2.8.4), and
## ‖(I−P
## L
) (f−I
## Ξ,L
f)‖
## H
τ
≤‖f−I
## Ξ,L
f‖
## N
## Φ,L
by (2.8.3).  Hence,
‖f−I
## Ξ,L
f‖
## H
σ
≤Ch
τ−σ
## Ξ
‖f−I
## Ξ,L
f‖
## H
σ
+Ch
τ−σ
## Ξ
‖f−I
## Ξ,L
f‖
## N
## Φ,L
## .
By Proposition 2.4.3,‖f−I
## Ξ,L
f‖
## N
## Φ,L
## ≤ ‖f‖
## N
## Φ,L
.  We can assumeh
## Ξ
is
small enough thatCh
τ−σ
## Ξ
## ≤1/2.  Thus,
‖f−I
## Ξ,L
f‖
## H
σ
## ≤
## 1
## 2
‖f−I
## Ξ,L
f‖
## H
σ
+Ch
τ−σ
## Ξ
## ‖f‖
## N
## Φ,L
## .
## Subtracting
## 1
## 2
‖f−I
## Ξ,L
f‖
## H
σ
from both sides and multiplying by 2 gives
‖f−I
## Ξ,L
f‖
## H
σ
≤Ch
τ−σ
## Ξ
## ‖f‖
## N
## Φ,L
=Ch
τ−σ
## Ξ
## ∞
## ∑
## `=L+1
## ̂
φ(`)
## −1
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
## ≤CC
## 2
h
τ−σ
## Ξ
## ∞
## ∑
## `=L+1
## (1 +λ
## `
## )
τ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
≤Ch
τ−σ
## Ξ
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
τ
## ∣
## ∣
## ∣
## ̂
f
## `
## ∣
## ∣
## ∣
## 2
=Ch
τ−σ
## Ξ
## ‖f‖
## H
τ
## .
(HereC
## 2
is the constant from (2.8.1).)
## 30

## Chapter 3
Kernel-Based Galerkin
## Methods
In this chapter we flush out the details for our kernel-based Galerkin meth-
ods.   In  section  3.1,  we  exhibit  a  quadrature  rule  that  will  be  used  to
approximate integrals that appear as entries of the stiffness matrix at the
heart of our Galerkin method.  It is a weighted sum of samples of the inte-
grand, analogous to Simpson’s rule or the trapezoid rule in Calculus.  The
error  estimate  for  our  quadrature  formula  is  a  direct  consequence  of  our
interpolation  error  estimates.   In  section  3.2,  we  introduce  the  Lagrange
basis, which is the basis for our approximation space in which we express
our  Galerkin  approximations.   In  section  3.3  we  give  the  details  for  the
Galerkin method.  Finally, we end the chapter by proving salient features
of the stiffness matrix in section 3.4.
We maintain all of our assumptions, both on the kernel Φ and the man-
ifoldM.  In particular, thatMis a compactC
## ∞
Riemannian manifold with
no boundary, and that our kernel Φ is of the form (2.2.1) with coefficients
## 31

that satisfy (2.8.1), either for all`if Φ is positive definite, or for` > Lif Φ
is conditionally positive definite with respect to Π
## L
## .
3.1    Kernel-Based Quadrature
In practice the integrals involved in our Galerkin method will not be cal-
culated directly.  Instead, we will employ aquadrature formula, which is a
weighted average of the values of the integrand at a denser set of centers.
Theorem 3.1.1.SupposeΦis  conditionally  positive  definite  with  respect
toΠ
## L
, and that (2.8.1) holds for someτ∈Nwithτ > d/2, and all` > L.
SupposeΞis a finiteΠ
## L
-unisolvent set of centers inM.  There is a unique
quadrature formula
## ∫
## M
fdμ≈
## ∫
## M
## I
## Φ,Ξ,L
fdμ=
## ∑
ξ∈Ξ
w
ξ
f(ξ) =:Q
## Ξ
## (f),
with weightsw={w
ξ
## }
ξ∈Ξ
, that is exact onV
## Ξ,L
and satisfies
## ∣
## ∣
## ∣
## ∣
## ∫
## M
f dμ−Q
## Ξ
## (f)
## ∣
## ∣
## ∣
## ∣
≤Ch
τ
## Ξ
## ‖f‖
## H
τ
forf∈H
τ
## .
Proof.Let{p
## 1
## ,...,p
## Q
}be  a  basis  for  Π
## L
.   Let1be  the  #Ξ×1  vector
whose entries are all 1, and let
## J
## 0
## =
## ∫
## M
## Φ(·,y)dμ,
which is independent of whichy∈Mwe choose.  Also, letJbe theQ×1
vector with entries
## J
k
## =
## ∫
## M
p
k
dμ
## 32

fork= 1,...,Q.  We claim that ifwandvare the (unique) #Ξ×1 and
Q×1 vectors that solve the system
## K
## Ξ,L
## 
## 
w
v
## 
## 
## =
## 
## 
## J
## 0
## 1
## J
## 
## 
## ,
then for alls∈V
## Ξ,L
## ,
## ∫
## M
s dμ=w
## T
s|
## Ξ
## =
## ∑
ξ∈Ξ
w
ξ
s(ξ).
Indeed, if
s=
## ∑
ξ∈Ξ
a
ξ
## Φ(·,ξ) +
## Q
## ∑
k=1
b
k
p
k
## ∈V
## Ξ,L
## ,
then  note  thataandbsatisfy  the  interpolation  equation  (2.4.2)  withc
replaced withs|
## Ξ
.  Thus we have
## ∫
## M
s dμ=
## ∑
ξ∈Ξ
a
ξ
## ∫
## M
## Φ(·,ξ)dμ+
## Q
## ∑
k=1
b
k
## ∫
## M
p
k
dμ
## =J
## 0
## 1
## T
a+J
## T
b=
## [
## J
## 0
## 1
## T
## J
## T
## ]
## 
## 
a
b
## 
## 
## =
## [
## J
## 0
## 1
## T
## J
## T
## ]
## K
## −1
## Ξ,L
## 
## 
s|
## Ξ
## 0
## 
## 
## =
## 
## 
## K
## −1
## Ξ,L
## 
## 
## J
## 0
## 1
## J
## 
## 
## 
## 
## T
## 
## 
s|
## Ξ
## 0
## 
## 
## =
## 
## 
w
v
## 
## 
## T
## 
## 
s|
## Ξ
## 0
## 
## 
## =w
## T
s|
## Ξ
## .
This proves the exactness onV
## Ξ,L
. The uniqueness of the weights follows
from the invertibility ofK
## Ξ,L
.  What remains to be shown is the error esti-
mate.  This follows directly from H ̈older’s inequality and the interpolation
## 33

error estimate from Theorem 2.8.5 withσ= 0:
## ∣
## ∣
## ∣
## ∣
## ∫
## M
f dμ−Q
## Ξ
## (f)
## ∣
## ∣
## ∣
## ∣
## ≤
## ∫
## M
|f−I
## Φ,Ξ,L
f|dμ
≤(Vol(M))
## 1/2
‖f−I
## Φ,Ξ,L
f‖
## L
## 2
≤Ch
τ
## Ξ
## ‖f‖
## H
τ
## .
There is a linear algebra technique, which can be found in [12, Section
2.2],  that  allows  for  numerically  efficient  computation  of  the  weightsw.
They are given byw=w
## ‖
## +w
## ⊥
, wherew
## ‖
## =P(P
## T
## P)
## −1
## Jandw
## ⊥
satisfies
the following system:
## 
## 
## K
## Ξ
## P
## P
## T
## 0
## 
## 
## 
## 
w
## ⊥
v
## 
## 
## =
## 
## 
## J
## 0
## 1−A
## Φ,Ξ
## P(P
## T
## P)
## −1
## J
## 0
## 
## 
## .
In  fact,w
## ⊥
can  be  obtained  without  having  to  solve  forv.   See  [12]  for
details.
## 3.2    The Lagrange Basis
There  are  obvious  bases  for  our  approximation  spaces.   If  Φ  is  positive
definite, then{Φ(·,ξ)}
ξ∈Ξ
is a basis forV
## Ξ
.  If Φ is conditionally positive
definite, then things are more complicated, because we can only allow cer-
tain  linear  combinations  of  these  “translates”,  and  we  have  to  include  a
basis for Π
## L
.  In both cases,  however,  there is an ideal basis to use:  the
Lagrange  basis.   For  eachξ∈Ξ,  we  find  the  unique  interpolantχ
ξ
that
interpolatesδ
ξ
## ={δ
ξη
## }
η∈Ξ
,  whereδ
ξη
=  1  ifξ=ηandδ
ξη
=  0  other-
wise.  Thus,χ
ξ
is the unique member of our approximation space for which
χ
ξ
## (η) =δ
ξη
.  TheLagrange basis, then, is{χ
ξ
## }
ξ∈Ξ
.  Computing the coeffi-
## 34

cientsα
ξ
## ={α
ξ,η
## }
ξ∈Ξ
andβ
ξ
## ={β
ξ,k
## }
k∈{1,...,Q}
of each Lagrange function
χ
ξ
## =
## ∑
η∈Ξ
α
ξ,η
## Φ(·,ξ) +
## Q
## ∑
k=1
β
ξ,k
p
k
can be computationally demanding; however, once computed interpolation
becomes immediate:
## I
## Φ,Ξ,L
f=
## ∑
ξ∈Ξ
f(ξ)χ
ξ
## .
Of particular interest is the decay of the Lagrange functions.  Bases that
have rapid decay away from where they are centered are desirable because
perturbations  in  data  only  affect  interpolants  locally.   In  the  Euclidean
setting,  Madych  and  Nelson,  in  [21],  considered  the  case  whereM=R
d
and Ξ =Z
d
.  Their Lagrange functions, in turns out, are all translates of a
single functionL
k
, which they referred to as “the fundamental function of
interpolation fork-harmonic splines”.  Byk-harmonic spline they mean, for
2k≥d+ 1, a functionffor whichf∈C
## 2k−d−1
and for which ∆
k
f= 0 on
## R
d
## \Z
d
.  Here ∆ is the usual Laplacian onR
d
.  They showed thatL
k
decays
exponentially  in  the  distance  from  the  origin.   The  kernel  from  Example
2.1.2 c is such a spline.  More generally, form∈Nwith 2m > d, thesurface
splines
## Φ(x,y) =
## 
## 
## 
## ‖x−y‖
## 2m−d
dodd,
## ‖x−y‖
## 2m−d
log‖x−y‖deven,
generate Lagrange functions that decay exponentially.
Another  author  to  investigate  the  cardinal  setting  is  Buhmann,  who
showed in [4] that for a large number of families of radial basis functions,
thecardinal  functionχdecays  algebraically  away  from  the  origin.   For
example, if we use the kernel from Example 2.1.2(c), but onR
## 3
instead of
onR
## 2
, we obtain|χ
ξ
(x)|≤C‖x‖
## −4
## 2
, but no faster.  Another way of putting
## 35

it is that lim
## ‖x‖
## 2
## →∞
## |χ
ξ
## (x)|‖x‖
## 4
## 2
is finite, but nonzero.  Yet another way is
thatχ
ξ
(x) =O
## (
## ‖x‖
## −4
## 2
## )
butχ
ξ
## (x)6=o
## (
## ‖x‖
## −4
## 2
## )
as‖x‖
## 2
→ ∞.  Again, the
Lagrange functions in this setting are all translates of a single function.
If  Φ  is  positive  definite  and  satisfies  (2.8.1)  withτ=m∈Nwith
τ > d/2, Proposition 2.4.4 gives thatχ
ξ
is the uniqueN
## Φ
norm-minimizing
elementuofN
## Φ
such  thatu|
## Ξ
## =δ
ξ
.   We  can  therefore  compareχ
ξ
to
a  bump  function  Ψ.   We  do  this  first  on  Euclidean  space,  and  then  we
will  use  the  metric  equivalence  from  Lemma  2.6.1  to  transfer  the  result
toM.  Letψ: [0,∞)→[0,1] satisfyψ∈C
## ∞
([0,∞)), suppψ= [0,1), and
ψ(0) = 1, and define Ψ :R
d
→[0,1] by setting Ψ(x) =ψ(‖x‖
## 2
## /q
## Ξ
## ).  Clearly
suppΨ =B(0,q
## Ξ
## ), Ψ∈W
m
## 2
## (R
d
), and  Ψ|
## Ξ
## =δ
ξ
, and it is routine to show
that‖Ψ‖
## W
m
## 2
## (R
d
## )
≤Cq
d
## 2
## −m
ξ
## .  Hence,
## ‖χ
ξ
## ‖
## W
m
## 2
≤Cq
d
## 2
## −m
## Ξ
## ,(3.2.1)
which we refer to from here on as ourbump estimate.
We consider two categories of kernels:  those whose Lagrange functions
decay algebraically away from their center, and those whose Lagrange func-
tions decay exponentially away from their center.  These two types of decay
can both be encapsulated in what are calledenergy estimates. In the sequel,
we will denote Euclidean balls of radiusrcentered atx∈R
d
byB(x,r),
and balls on the manifold of radiusrcentered atp∈Mbyb(p,r).
Definition 3.2.2.Suppose Φ is a kernel which satisfies (2.8.1) forτ=m∈
Nwithm > d/2, for all`∈Nif Φ is positive definite or for all` > L∈N
if Φ is conditionally positive definite with respect to Π
## L
## .
We say that Φ provides analgebraic energy estimateif there are positive
constantsCsuch that for a set of centers Ξ, the Lagrange basis{χ
ξ
## }
ξ∈Ξ
## 36

satisfies, forh
## Ξ
sufficiently small and 0≤r≤diam(M),
## ‖χ
ξ
## ‖
## W
m
## 2
## (b(ξ,r)
c
## )
≤Cq
d
## 2
## −m
## Ξ
## (
## 1 +
r
h
## Ξ
## )
## −2m
## .(3.2.3)
We say Φ provides anexponential energy estimateif there are positive
constantsCandνsuch  that  for  a  set  of  centers  Ξ,  the  Lagrange  basis
## {χ
ξ
## }
ξ∈Ξ
satisfies, forh
## Ξ
sufficiently small and 0≤r≤diam(M),
## ‖χ
ξ
## ‖
## W
m
## 2
## (b(ξ,r)
c
## )
≤Cq
d
## 2
## −m
## Ξ
exp
## (
## −ν
r
h
## Ξ
## )
## .(3.2.4)
We note that both energy estimates withr= 0 agree with our bump
estimate.   Energy  estimates  will  allow  us  to  obtain  pointwise  bounds  on
χ
ξ
(x) and∇χ
ξ
(x) that depend only on the distance betweenxandξ, but
first,  let us elaborate on the situation described in section 2.3,  where we
considered kernels whose coefficients were reciprocals of polynomials inλ
## `
## .
These types of kernels, it turns out, provide energy estimates.
Definition 3.2.5.Letm∈Nwithm > d/2.  We call Φ
m
## :M×M→R
polyharmonicif it is of the form (2.2.1) with coefficients
## ̂
φ
m
(`) =Q(λ
## `
## )
## −1
## ,
whereQ(x) =
## ∑
m
k=0
c
k
x
k
is a polynomial of degreemwithc
m
## >0.  Corre-
sponding to a polyharmonic kernel Φ
m
is a differential operatorL
m
## =Q(∆)
of order 2mfor which Φ
m
is the fundamental solution.  When we write that
## Φ
m
is polyharmonic, it is understood that the degree of the polynomialQ
is captured in the indexm.
Note that the conditionc
m
>0 means that
## ̂
φ(`) =Q(λ
## `
## )
## −1
>0 for all
`larger than someL∈N.  Hence, Φ
m
is conditionally positive definite with
respect to Π
## L
.  The results in [16, Section 5] say that if, in addition,L
m
annihilates the auxiliary space Π
## L
, then Φ
m
provides an exponential energy
## 37

estimate.  Obviously, then, if
## ̂
φ
m
(`)>0 for all`∈N, then Φ
m
is positive
definite and thus also provides an exponential energy estimate, since in this
case the auxiliary space is trivial.  If Φ
m
is conditionally positive definite
with  respect  to  Π
## L
andL
m
doesnotannihilate  the  auxiliary  space  Π
## L
## ,
then it is only known that Φ
m
provides an algebraic energy estimate.  As
the authors in [16] mention,  while only an algebraic energy estimate has
been proven in this case, there is sufficient cause to believe that this is just
an  artifact  of  the  proof.   In  any  case,  we  will  examine  both  possibilities,
algebraic and exponential.
Note  also  that  if  Φ
m
is  polyharmonic,  then  the  coefficients
## ̂
φ
m
## (`)  =
## Q(λ
## `
## )
## −1
satisfy  (2.8.1)  withτ=mand` > L,  since  bothQ(λ
## `
## )
## −1
and
## (1 +λ
## `
## )
## −m
are reciprocals of polynomials inλ
## `
of degreemwith positive
leading coefficients.
To recapitulate, ifMis two-point homogeneous and Φ
m
is polyharmonic,
then Φ
m
provides an energy estimate andN
## Φ
m
## =H
m
.  In general, this is
not the case.  There are also kernels Φ
m
that provide energy estimates with
parametermthat are not polyharmonic.  This thesis treats those as well.
We  note  that  our  definition  of  a  kernel  providing  an  energy  estimate
includes  the  assumption  that  (2.8.1)  holds  fort=m∈N,  at  least  for`
large enough, and thatm > d/2.  As such,N
## Φ
m
## =H
m
, and we will not
repeat the assumptionm > d/2.
Proposition 3.2.6.SupposeΦ
m
provides an energy estimate,Ξis a finite
subset  ofM(Π
## L
-unisolvent  ifΦ
m
is  conditionally  positive  definite  with
respect toΠ
## L
), and{χ
ξ
## }
ξ∈Ξ
is the corresponding Lagrange basis.
IfΦ
m
provides  an  algebraic  energy  estimate  as  in(3.2.3),  then  forh
## Ξ
sufficiently small,
## |χ
ξ
(p)|≤C
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## .
## 38

IfΦprovides an exponential energy estimate as in(3.2.4), then forh
## Ξ
sufficiently small,
## |χ
ξ
(p)|≤Cexp
## (
## −ν
dist(p,ξ)
h
## Ξ
## )
## .
Proof.Letr= dist(p,ξ).  First, we have|χ
ξ
## (p)|≤‖χ
ξ
## ‖
## L
## ∞
## (b(ξ,r)
c
## )
.  We now
use a Zeros Lemma, but not the one from Lemma 2.7.1, which only applies
to the whole space.  We use instead the Zeros Lemma from [16, Theorem
A.11], which applies to complements of balls.  Thus,
## |χ
ξ
(p)|≤Ch
m−
d
## 2
## Ξ
## ‖χ
ξ
## ‖
## W
m
## 2
## (b(ξ,r)
c
## )
## ,
and the results now follow from the energy estimates.
A similar proof, but using [16, Corollary A.15] instead of [16, Theorem
A.11], gives the following.
Proposition 3.2.7.SupposeΦ
m
provides an energy estimate,Ξis a finite
subset  ofM(Π-unisolvent  ifΦ
m
is  conditionally  positive  definite  with  re-
spect toΠ), and{χ
ξ
## }
ξ∈Ξ
is the corresponding Lagrange basis.  IfΦprovides
an algebraic energy estimate as in(3.2.3), then forh
## Ξ
sufficiently small,
## |∇χ
ξ
(p)|≤Ch
## −1
## Ξ
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## .
IfΦprovides an exponential energy estimate as in(3.2.4), then
## |∇χ
ξ
(p)|≤Ch
## −1
## Ξ
exp
## (
## −ν
dist(p,ξ)
h
## Ξ
## )
## .
The decay of the Lagrange functions has been used to prove the following
stability bound, which we refer to astheL
## 2
-stability of the Lagrange basis.
It says that the`
## 2
norm of a vector can be controlled by theL
## 2
norm of the
## 39

corresponding linear combination of Lagrange functions, and vice versa.
Theorem 3.2.8.([16, Theorem 5.7])SupposeΦ
m
provides an energy esti-
mate,Ξis a finite subset ofM(Π-unisolvent ifΦ
m
is conditionally positive
definite with respect toΠ), and{χ
ξ
## }
ξ∈Ξ
is the corresponding Lagrange basis.
There exist constantsC
## 1
andC
## 2
such that, forh
## Ξ
sufficiently small,
## C
## 1
h
d/p
## Ξ
## ‖β‖
## `
## 2
## (Ξ)
## ≤
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
β
ξ
χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## (M)
## ≤C
## 2
h
d/p
## Ξ
## ‖β‖
## `
## 2
## (Ξ)
holds for allβ={β
ξ
## }
ξ∈Ξ
## ∈`
## 2
## (Ξ).
We end this section by noting a connection between the Lagrange func-
tions and our quadrature rule; namely, that the weights are the integrals of
the Lagrange functions.  This is becauseχ
ξ
is in the approximation space,
and hence the quadrature rule is exact:
## ∫
## M
χ
ξ
dμ=w
## T
χ
ξ
## |
## Ξ
## =w
## T
δ
ξ
## =w
ξ
## .
While this is a useful theoretical fact, and one that we will use later when
showing  the  off-diagonal  decay  of  our  quadratized  stiffness  matrix,  it  is
important to stress that this hasnothingto do with how the weights are
actually computed.  They are computed using the linear algebra technique
discussed at the end of section 3.1
3.3    Kernel-Based Galerkin Methods
We will be approximating weak solutions to a differential equation of the
formLu=f, withLgiven indivergence formby
## Lu=−∇
## ∗
a(∇u,·) +bu.(3.3.1)
## 40

Let us first be more precise about what this means.  We start with a sym-
metric rank (1,1) tensora.  This means that for eachp∈M,
a(p) :T
## ∗
p
## M×T
p
## M−→R.
Iff∈C
## ∞
(M),  andp∈M,  then∇f(p) is a rank 1 covariant tensor;  i.e.
∇f(p) :T
p
M→R,v7→ ∇
v
f(p),  which is a directional derivative.  If we
put this as the first argument ina(p) and leave the second open,  we get
another rank 1 covariant tensor,
a(p) (∇f(p),·) :T
p
## M−→R.
Being a rank 1 covariant tensor, we can apply∇
## ∗
to it to obtain a scalar,
## ∇
## ∗
a(∇f,·).   This  is  what  is  meant  in  the  principle  part  of  (3.3.1),  the
coordinate-free expression of our differential equation.
In coordinates (U,φ), whereφ(U) =U⊂R
d
andφ(p) = (x
## 1
## ,...,x
d
## )∈
U, lete
## 1
## ,...,e
d
be a basis forT
p
## Mande
## 1
## ,...,e
d
the dual basis forT
## ∗
p
## M.
## Then∇f(p) =
## ∑
j
## ∂f
## ∂x
j
e
j
anda(p) =
## ∑
j,k
a
k
j
e
k
## ⊗e
j
, and so
a(p) (∇f(p),·) =
## ∑
j,k
a
k
j
e
k
## ⊗e
j
## (
## ∑
## `
## ∂f
## ∂x
## `
e
## `
## ,·
## )
## =
## ∑
j,k
a
k
j
## ∂f
## ∂x
k
e
j
## .
By the definition of∇
## ∗
, for any covariant rank 1 tensorT,
## ∫
## M
〈∇f,T〉
p
dμ(p) =
## ∫
## M
f(p)∇
## ∗
## T(p)dμ(p).
Recall that ifS=
## ∑
j
## S
j
e
j
andT=
## ∑
k
## T
k
e
k
, then
## 〈S,T〉
p
## =
## ∑
j,k
## S
j
## T
k
## 〈
e
j
## ,e
k
## 〉
p
## =
## ∑
j,k
## S
j
## T
k
g
jk
## .
## 41

So, in a coordinate neighborhoodU,
## ∫
## U
## 〈S,T〉
p
dμ(p) =
## ∫
## U
## ∑
j,k
## S
j
## T
k
g
jk
## (x)
## √
det(g(x))dx.
ReplacingSwith∇f, and assuming that the closure of the support offis
contained inU, we have
## ∫
## U
〈∇f,T〉
p
dμ(p) =
## ∫
## U
## ∑
j,k
## ∂f
## ∂x
j
## T
k
g
jk
## (x)
## √
det(g(x))dx
## =−
## ∫
## U
f(φ(x))
## ∑
j,k
## ∂
## ∂x
j
## [
## T
k
g
jk
## (x)
## √
det(g(x))
## ]
dx.
Multiplying and dividing by
## √
det(g(x)) gives
## ∫
## U
f(p)∇
## ∗
## T(p)dμ(p)
## =−
## ∫
## U
f(φ(x))
## 1
## √
det(g(x))
## ∑
j,k
## ∂
## ∂x
j
## [
## T
k
g
jk
## (x)
## √
det(g(x))
## ]
## √
det(g(x))dx.
This shows that, in coordinates,
## ∇
## ∗
## T=−
## 1
## √
det(g(x))
## ∑
j,k
## ∂
## ∂x
j
## [
## T
k
g
jk
## √
det(g(x))
## ]
## .
IfT=a(p) (∇f(p),·),  thenT
k
## =
## ∑
## `
a
l
k
## ∂f
## ∂x
## `
,  and  so,  in  coordinates,  our
differential equation is
## Lu=
## 1
## √
det(g)
d
## ∑
j,k,`=1
## ∂
## ∂x
j
## [
g
jk
## √
det(g)a
## `
k
## ∂u
## ∂x
## `
## ]
## +bu=f.(3.3.2)
## 42

If we multiply the principle part of (3.3.1) byv∈H
## 1
and integrate, we
get
## ∫
## M
v(p) (∇
## ∗
a(p) (∇u(p),·))dμ(p) =
## ∫
## M
## 〈a(∇u,·),∇v〉
p
dμ(p)
## =
## ∫
## M
a
## ]
## (p) (∇u(p),∇v(p))dμ(p).
Thus, the weak form of (3.3.1) is
## 〈u,v〉
a,b
## :=
## ∫
## M
## (
a
## ]
## (∇u,∇v) +buv
## )
dμ=
## ∫
## M
fv dμ=:λ
f
## (v).(3.3.3)
We note three things.  First, that the components ofa
## ]
area
ij
## =
## ∑
k
g
jk
a
i
k
## .
Next,  in  the  casea
## [
=gandb=  0  the  equation  reduces  to−∆u=f.
## Finally,〈u,v〉
a,b
=〈Lu,v〉
## L
## 2
## .
We assume there are constants 0< b
## 1
## ≤b
## 2
such thatb
## 1
## ≤b(p)≤b
## 2
for
allp∈M.  We also assume thata
## ]
is symmetric and positive definite in the
sense that there are constants 0< a
## 1
## ≤a
## 2
such that
a
## 1
## 〈v,v〉
p
## ≤a
## ]
## (p)(v,v)≤a
## 2
## 〈v,v〉
p
for eachp∈Mandv∈T
## ∗
p
M.  Finally, we assume thata
## ]
is bounded in the
sense that
## ∣
## ∣
a
## ]
## (v,w)
## ∣
## ∣
≤C|v||w|
for allv,w∈T
## ∗
p
## M.
The conditions onaandbensure that〈·,·〉
a,b
is bounded and coercive,
andλ
f
is bounded.  Hence, the Lax-Milgrim Theorem applies, and we have
the following.
Proposition 3.3.4.The bilinear form〈·,·〉
a,b
is coercive and bounded on
## H
## 1
and defines an inner product onH
## 1
, and the norms‖·‖
a,b
and‖·‖
## H
## 1
## 43

are  equivalent.  In  addition,  iff∈L
## 2
there  is  a  uniqueu∈H
## 1
such  that
## 〈u,v〉
a,b
## =λ
f
(v)for allv∈H
## 1
## .
The next proposition holds on coordinate patches by [9, pp.  261-269].
SinceMis compact, it holds globally.
Proposition  3.3.5.Ifuis  the  weak  solution  toLu=f,  withf∈H
s
## ,
s≥0, then there is a constantCthat is independent ofuandfsuch that
## ‖u‖
## H
s+2
≤C‖f‖
## H
s
## .
Theweak solutionutoLu=fis the solution to the following problem:
givenf∈L
## 2
, findu∈H
## 1
such that〈u,v〉
a,b
## =λ
f
(v) for allv∈H
## 1
## .  The
Galerkin approximation scheme modifies this problem to read: givenf∈L
## 2
findu
## Ξ
## ∈V
## Ξ,L
such that〈u
## Ξ
## ,v〉
a,b
## =λ
f
(v) for allv∈V
## Ξ,L
.  TheGalerkin
approximationu
## Ξ
to the solutionuofLu=fis obtained as follows.  The
stiffness matrixin the Lagrange basis,B
## Ξ
## ={B
ξη
## }
ξ,η∈Ξ
, has entries
## B
ξη
## =〈χ
ξ
## ,χ
η
## 〉
a,b
The Galerkin approximation, then, is
u
## Ξ
## =
## ∑
ξ∈Ξ
γ
ξ
χ
ξ
## ,
where the coefficientsγ={γ
ξ
## }
ξ∈Ξ
are obtained by solving the linear system
## B
## Ξ
γ=ω, withω={ω
ξ
## }
ξ∈Ξ
the vector with entriesω
ξ
## =λ
f
## (χ
ξ
## ).
By construction,〈u
## Ξ
## ,χ
ξ
## 〉
a,b
## =λ
f
## (χ
ξ
) for eachξ∈Ξ; thus,〈u
## Ξ
## ,v〉
a,b
## =
λ
f
(v) for eachv∈V
## Ξ,L
.  Sinceuis the weak solution toLu=f,  surely
## 〈u,v〉
a,b
## =λ
f
## (v).  Hence〈u−u
## Ξ
## ,v〉
a,b
= 0 for allv∈V
## Ξ,L
, and so we see
thatu
## Ξ
is the orthogonal projection ofuontoV
## Ξ,L
in the inner product
## 〈·,·〉
a,b
## .
We now give an error estimate for Galerkin approximation.  We state
## 44

it  in  the  case  of  a  conditionally  positive  definite  kernel.   Since  a  positive
definite kernel is conditionally positive definite with respect to any auxiliary
space, it holds for positive definite kernels as well.
Theorem 3.3.6.SupposeΦ
m
provides an energy estimate andΞis a finite
subset ofM(Π-unisolvent ifΦ
m
is conditionally positive with respect toΠ
## ).   Letu
## Ξ
be  the  corresponding  Galerkin  approximation  to  the  solution  of
Lu=f,  wheref∈H
m−2
.   There  exists  a  positive  constantCthat  is
independent offandΞsuch that, forh
## Ξ
sufficiently small,
## ‖u−u
## Ξ
## ‖
## H
## 1
≤Ch
m−1
## Ξ
## ‖f‖
## H
m−2
## .
Proof.By  the  equivalence  of  the  norms‖·‖
## H
## 1
and‖·‖
a,b
,  and  because
u
## Ξ
is  the  orthogonal  projection  ofuontoV
## Ξ,L
with  respect  to  the  inner
product〈·,·〉
a,b
## ,
## ‖u−u
## Ξ
## ‖
## H
## 1
≤C‖u−u
## Ξ
## ‖
a,b
=Cmin
v∈V
## Ξ,L
## ‖u−v‖
a,b
≤C‖u−I
## Ξ,L
u‖
a,b
## .
Using  that  norm  equivalence  again  and  Theorem  2.8.5  withτ=mand
σ= 1 gives
## ‖u−u
## Ξ
## ‖
## H
## 1
≤C‖u−I
## Ξ,L
u‖
## H
## 1
≤Ch
m−1
## Ξ
## ‖u‖
## H
m
## .
Finally, by Proposition 3.3.5,
## ‖u−u
## Ξ
## ‖
## H
## 1
≤Ch
m−1
## Ξ
## ‖f‖
## H
m−2
## .
## 45

3.4    The Stiffness Matrix in the Lagrange Ba-
sis
It  is  not  necessary  to  use  the  Lagrange  basis  to  compute  the  Galerkin
approximation;  however,  one  of  the  advantages  is  that  the  entries  of  the
stiffness matrix decay away from the main diagonal.  This makes computing
the  Galerkin  approximation  numerically  stable.   We  illustrate  this  decay
now.
Lemma  3.4.1.SupposeΦ
m
provides  an  energy  estimate,Ξis  a  finite
set  of  centers  inM(Π-unisolvent  ifΦ
m
is  conditionally  positive  definite
with respect toΠ), andB
## Ξ
is the stiffness matrix in the Lagrange basis for
Galerkin approximation using the kernelΦ
m
and centersΞ.
IfΦ
m
provides  an  algebraic  energy  estimate  as  in(3.2.3),  then  there
is  a  positive  constantCdepending  only  onΦ
m
andρsuch  that,  forh
## Ξ
sufficiently small,
## |B
ξη
|≤Ch
d−2
## Ξ
## (
## 1 +
dist(ξ,η)
h
## Ξ
## )
## −2m
## .
IfΦ
m
provides an exponential energy estimate as in(3.2.4), then there
is  a  positive  constantCdepending  only  onΦ
m
andρsuch  that,  forh
## Ξ
sufficiently small,
## |B
ξη
|≤Ch
d−2
## Ξ
exp
## (
## −ν
dist(ξ,η)
h
## Ξ
## )
## .
Proof.First suppose Φ
m
provides an algebraic energy estimate.  By Propo-
## 46

sition 3.2.6,
## ∣
## ∣
## ∣
## ∣
## ∫
## M
bχ
ξ
χ
η
dμ
## ∣
## ∣
## ∣
## ∣
≤Cb
## 2
## ∫
## M
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(p,η)
h
## Ξ
## )
## −2m
dμ(p).
## (3.4.2)
We show a similar bound holds for the principle part.  SinceMis compact,
the open cover{b(q,r
## M
## /3)}
q∈M
has a finite subcover
## {
b(q
i
## ,r
## M/3
## )
## }
n
i=1
## .  Let
## Ω
## 1
## =b(q
## 1
## ,r
## M
/3) and fori= 2,...,nset
## Ω
i
## =b(q
i
## ,r
## M
## /3)\
i−1
## ⋃
j=1
## Ω
j
## .
ThenM=
## ⋃
n
i=1
## Ω
i
is a disjoint union.  Note that each Exp
i
## = Exp
q
i
is a
bijective isometry from Ω
i
toU
i
## = Exp
## −1
i
## (Ω
i
), and so
## ∣
## ∣
## ∣
## ∣
## ∫
## M
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )dμ
## ∣
## ∣
## ∣
## ∣
## =
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
n
## ∑
j=1
## ∫
## Ω
i
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )dμ
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## =
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
n
## ∑
i=1
## ∫
## U
i
## ∑
j,k
a
jk
## (x)
## ∂χ
ξ
## ∂x
j
## (x)
## ∂χ
η
## ∂x
k
## (x)dx
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ≤C
## ∑
i,j,k
## ∫
## U
i
## ∣
## ∣
## ∣
## ∣
## ∂χ
ξ
## ∂x
j
## (x)
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ∂χ
η
## ∂x
k
## (x)
## ∣
## ∣
## ∣
## ∣
dx,
where  we  have  abused  notation  slightly  by  writing,  for  example,
## ∂χ
ξ
## ∂x
j
## (x)
instead  of
## ∂χ
ξ
◦Exp
i
## ∂x
j
## (x).   Surely
## ∣
## ∣
## ∣
## ∂χ
ξ
## ∂x
j
## (x)
## ∣
## ∣
## ∣
## ≤ |∇χ
ξ
(x)|,  and  so  by  Theorem
## 3.2.7,
## ∣
## ∣
## ∣
## ∣
## ∫
## M
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )dμ
## ∣
## ∣
## ∣
## ∣
≤Ch
## −2
## Ξ
## ∫
## M
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(p,η)
h
## Ξ
## )
## −2m
dμ(p).
## (3.4.3)
We now split up the integrals in (3.4.2) and (3.4.3) onto two half spaces.
## 47

## Let
## Ω
ξ
={x∈M:  dist(x,ξ)<dist(x,η)}
and
## Ω
η
={x∈M:  dist(x,η)<dist(x,ξ)}.
## Then Ω
ξ
## ∪Ω
η
is, minus a set of measure zero,M, and Ω
ξ
## ∩Ω
η
## =∅.  Hence,
for an integrable functionf,
## ∫
## M
f dμ=
## ∫
## Ω
ξ
f dμ+
## ∫
## Ω
η
f dμ.
Note that forx∈Ω
ξ
, dist(x,η)≥
## 1
## 2
dist(ξ,η), and so
## ∫
## Ω
ξ
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(p,η)
h
## Ξ
## )
## −2m
dμ(p)
## ≤C
## (
## 1 +
dist(ξ,η)
h
## Ξ
## )
## −2m
## ∫
## M
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p).
## Now,
## ∫
## M
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p)
## =
## ∫
b(ξ,r
## M
## )
## (
## 1 +
dist(p,ξ)
h
ξ
## )
## −2m
dμ(p)
## +
## ∫
b(ξ,r
## M
## )
c
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p).
Forp∈Mwith dist(p,ξ)≥r
## M
## ,
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## ≤r
## −2m
## M
h
## 2m
## Ξ
## ,
## 48

and so
## ∫
b(ξ,r
## M
## )
c
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p)≤Vol(M)r
## −2m
## M
h
## 2m
## Ξ
≤Ch
d
## Ξ
## .
Lemma  2.6.1  applied  tog(p)  =
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −m
withj=  0  and  Ω  =
## Exp
## −1
ξ
## (b(ξ,r
## M
)) gives
## ∫
b(ξ,r
## M
## )
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p) =‖g‖
## 2
## L
## 2
## (b(ξ,r
## M
## ))
≤C‖g◦Exp‖
## 2
## L
## 2
(Exp
## −1
ξ
## (b(ξ,r
## M
## )))
## ≤C
## ∫
## R
d
## (
## 1 +
## ‖x‖
## 2
h
## Ξ
## )
## −2m
dx=C
## ′
h
d
## Ξ
## ,
where we have used the integrability of (1+‖x‖
## 2
## )
## −2m
.  The same arguments
apply to the integral over Ω
η
, and therefore
## |B
ξη
## |≤C
## ′
h
d−2
## Ξ
## (
## 1 +
dist(ξ,η)
h
ξ
## )
## −2m
## .
The result in the case that Φ
m
provides an exponential energy estimate is
proved in an almost identical way.
Lemma 3.4.4.SupposeΦ
m
provides an energy estimate,Ξis a finite set
of  centers  inM(Π-unisolvent  ifΦ
m
is  conditionally  positive  definite  with
respect toΠ), and{χ
ξ
## }
ξ∈Ξ
is the corresponding Lagrange basis.  LetB
## Ξ
be
the  stiffness  matrix  in  the  Lagrange  basis.  There  is  a  positive  constantC
depending only onΦ
m
andρsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
≤Ch
## −d
## Ξ
## .
## 49

Proof.It suffices to show thatλ
min
## (B
## Ξ
)≥Cq
d
## Ξ
, for then
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
≤Cq
## −d
## Ξ
## =
## Cρ
d
h
## −d
## Ξ
.   For  that  it  suffices  to  show  thatv
## T
## B
## Ξ
v≥Cq
d
## Ξ
## ‖v‖
## 2
## `
## 2
## (Ξ)
for  all
vectorsv={v
ξ
## }
ξ∈Ξ
.  Given such av,  letu=
## ∑
ξ∈Ξ
v
ξ
χ
ξ
,  so that∇u=
## ∑
ξ∈Ξ
v
ξ
## ∇χ
ξ
.  We have
v
## T
## B
## Ξ
v=
## ∑
ξ∈Ξ
## ∑
η∈Ξ
v
ξ
v
η
## ∫
## M
## (
a
## ]
## (∇χ
ξ
## ,∇χ
η
## ) +bχ
ξ
χ
η
## )
dμ
## =
## ∫
## M
## 
## 
a
## ]
## 
## 
## ∑
ξ∈Ξ
v
ξ
## ∇χ
ξ
## ,
## ∑
η∈Ξ
v
η
## ∇χ
η
## 
## 
## +b
## 
## 
## ∑
ξ∈Ξ
v
ξ
χ
ξ
## 
## 
## 
## 
## ∑
η∈Ξ
v
η
χ
η
## 
## 
## 
## 
dμ
## =
## ∫
## M
## (
a
## ]
## (∇u,∇u) +bu
## 2
## )
dμ.
The metric tensorabeing positive definite ensuresa
## ]
## (∇u,∇u)>0.  Hence
we can use theL
## 2
stability of the Lagrange basis to obtain
v
## T
## B
## Ξ
v≥
## ∫
## M
bu
## 2
dμ≥b
## 1
## ‖u‖
## 2
## L
## 2
## =b
## 1
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
v
ξ
χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
## ≥C
## 2
## 1
b
## 1
q
d
## Ξ
## ‖v‖
## 2
## `
## 2
## (Ξ)
## .
## 50

## Chapter 4
Quadratized Kernel-Based
## Galerkin Approximation
In this chapter we explore what happens when we replace every integral in
our Galerkin approximation scheme with the corresponding quadrature ap-
proximation.  In Section 4.1 we introduce the quadratized stiffness matrix
and establish bounds on the entries and norms of the quadratized stiffness
matrix  and  the  difference  between  the  stiffness  matrix  and  the  quadra-
tized stiffness matrix.  In Section 4.2 we prove an error estimate between
the Galerkin approximation and the quadratized Galerkin approximation
in Theorem 4.2.1, which immediately translates into an error estimate be-
tween  the  weak  solution  and  the  quadratized  Galerkin  approximation  in
Corollary 4.2.7.  We end this chapter by detailing algorithms for computing
the quadratized stiffness matrix.
## 51

## 4.1    The Quadratized Stiffness Matrix
Thequadratized  stiffness  matrixis the matrix obtained from the stiffness
matrix by replacing each entry, which is an integral, by a quadrature es-
timate of that integral.  The kernel used for quadrature is the same,  but
the  centers  used  for  quadrature  are  denser  than  the  centers  used  for  the
Galerkin approximation.
## Suppose  Φ
m
and  Ξ  are  the  kernel  and  centers,  respectively,  used  for
Galerkin approximation. Let Λ be centers used for quadrature. The quadra-
tized stiffness matrix,B
## Λ
## Ξ
, has entriesB
## Λ
ξη
## =Q
## Λ
## (
a
## ]
## (∇χ
ξ
## ,∇χ
η
## ) +bχ
ξ
χ
η
## )
## .
We make further assumptions onm, Ξ, and Λ, which are listed in Lemma
## 4.1.1.
The  main  objective  of  this  section,  Lemmas  4.1.16  and  4.1.21,  is  to
bound  the  2-norm  of  the  difference  between  the  stiffness  matrix  and  the
quadratized stiffness matrix.  To that end we will need the following propo-
sitions.  Lemma 4.1.1 gives a bound on the entries of the quadratized stiff-
ness  matrix  as  a  function  of  their  distance  from  the  diagonal.   Theorem
4.1.5  gives  a  bound  on  the  Sobolev  norms  of  the  principle  part  and  the
non-principle part of the weak form of the equationLu=fwhen the in-
puts  are  Lagrange  functions.   Both  are  used  after  quadrature  is  applied
when quadratizing the stiffness matrix.  The theorem is modeled after [22,
Theorem 4.4], though the differences in the proof are significant.  Lemma
4.1.14 gives a bound on the entries of the difference between the stiffness
matrix  and  the  quadratized  stiffness  matrix.   Lemmas  4.1.16  and  4.1.21
bound the 2-norm of that difference.  We then have Lemma 4.1.25, which
controls the 2-norm of the inverse of the quadratized stiffness matrix.  It
includes a threshold upper bound forh
## Λ
in terms ofh
## Ξ
in order to work.
As  will  be  discussed,  there  is  an  alternative  way  to  achieve  the  result  in
## 52

Lemma 4.1.25 if we know that the quadrature weights are positive.  This is
done in Lemma 4.2.12.
In what follows whenever we say “there is a positive constantC” it is
understood thatCdepends on Φ
m
,ρ,  and the parametersaandbfrom
our differential equation (3.3.1).  If other dependencies are specified, those
are in addition to those just mentioned.  We also note that the condition
m > d/2 is included in Definition 3.2.2, and as such will not be repeated.
Lemma 4.1.1.SupposeΦ
m
provides an energy estimate,Ξis a finite set
of  centers  inM(Π-unisolvent  ifΦ
m
is  conditionally  positive  definite  with
respect  toΠ),  and{χ
ξ
## }
ξ∈Ξ
is  the  corresponding  Lagrange  basis.  LetΛbe
another set of centers inM.  Assumem≥2,#Ξ<#Λ, andh
## Λ
< h
## Ξ
## .  Let
## B
## Λ
## Ξ
be the quadratized stiffness matrix.
IfΦ
m
provides an algebraic energy estimate as in(3.2.3), then there is
a positive constantCsuch that, forh
## Ξ
sufficiently small,
## ∣
## ∣
## B
## Λ
ξη
## ∣
## ∣
≤Ch
d−2
## Ξ
## (
## 1 +
dist(ξ,η)
h
## Ξ
## )
## −2m
## .
IfΦ
m
provides an exponential energy estimate as in(3.2.4), then there
is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∣
## ∣
## B
## Λ
ξη
## ∣
## ∣
≤Ch
d−2
## Ξ
exp
## (
## −ν
dist(ξ,η)
h
## Ξ
## )
## .
## 53

Proof.First, we have by Theorems 3.2.6 and 3.2.7 that
## ∣
## ∣
## B
## Λ
ξη
## ∣
## ∣
## =
## ∣
## ∣
## ∣
## ∣
## ∣
## ∑
λ∈Λ
## (
a
## ]
## (λ) (∇χ
ξ
## ,∇χ
η
## ) +b(λ)χ
ξ
## (λ)χ
η
## (λ)
## )
w
λ
## ∣
## ∣
## ∣
## ∣
## ∣
## ≤C
## ∑
λ∈Λ
## (|∇χ
ξ
## (λ)||∇χ
η
## (λ)|+|χ
ξ
## (λ)||χ
η
## (λ)|)|w
λ
## |
≤Ch
## −2
## Ξ
## ∑
λ∈Λ
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(λ,η)
h
## Ξ
## )
## −2m
## |w
λ
## |.
## Let{ ̃χ
λ
## }
λ∈Λ
be the Lagrange basis for Φ
m
and Λ.  By theL
## 1
stability of
this basis,
## |w
λ
## |=
## ∣
## ∣
## ∣
## ∣
## ∫
## M
## ̃χ
λ
dμ
## ∣
## ∣
## ∣
## ∣
## ≤‖ ̃χ
λ
## ‖
## L
## 1
≤Cq
d
## Λ
## .
## Hence,
## ∣
## ∣
## B
## Λ
ξη
## ∣
## ∣
≤Ch
## −2
## Ξ
q
d
## Λ
## ∑
λ∈Λ
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(λ,η)
h
## Ξ
## )
## −2m
## .(4.1.2)
## Let
## Λ
ξ
={λ∈Λ :  dist(λ,ξ)<dist(λ,η)}
and
## Λ
η
={λ∈Λ :  dist(λ,η)≤dist(λ,ξ)}.
Note that forλ∈Λ
ξ
, dist(λ,η)≥
## 1
## 2
dist(ξ,η) and so
## ∑
λ∈Λ
ξ
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## (
## 1 +
dist(λ,η)
h
ξ
## )
## −2m
## ≤C
## (
## 1 +
dist(ξ,η)
h
ξ
## )
## −2m
## ∑
λ∈Λ
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## .
## (4.1.3)
## 54

Note that ifp∈b(λ,q
## Λ
), then dist(λ,ξ)≥
## 1
## 2
dist(p,ξ), and hence
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## ≤C
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
## .
## Therefore,
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## =
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
## ·q
## −d
## Λ
## ∫
b(λ,q
## Λ
## )
dμ
≤Cq
## −d
## Λ
## ∫
b(λ,q
## Λ
## )
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p).
Since the collection of balls of radiusq
## Λ
centered atλ∈Λ are disjoint, we
have
## ∑
λ∈Λ
## (
## 1 +
dist(λ,ξ)
h
## Ξ
## )
## −2m
≤Cq
## −d
## Λ
## ∑
λ∈Λ
## ∫
b(λ,q
## Λ
## )
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p)
≤Cq
## −d
## Λ
## ∫
## M
## (
## 1 +
dist(p,ξ)
h
## Ξ
## )
## −2m
dμ(p)≤Cq
## −d
## Λ
h
d
## Ξ
## ,
## (4.1.4)
since  the  integral  in  the  last  line  of  (4.1.4)  is  precisely  the  integral  we
showed was bounded byCh
d
## Ξ
in the proof of Lemma 3.4.1.  Putting (4.1.4)
into (4.1.3) and then putting that into (4.1.2) yields the result.  The result
in the case that Φ
m
provides an exponential energy estimate is proven in
an almost identical way.
Theorem 4.1.5.SupposeΦ
m
provides an energy estimate with parameter
m > d/2 + 1,Ξis  a  finite  subset  ofM(Π-unisolvent  ifΦ
m
is  condition-
ally  positive  definite  with  respect  toΠ),  and{χ
ξ
## }
ξ∈Ξ
is  the  corresponding
Lagrange basis.  Thenbχ
ξ
χ
η
## ∈H
m
## ∩L
## ∞
anda
## ]
## (∇χ
ξ
## ,∇χ
η
## )∈H
m−1
; more-
## 55

over, there is a positive constantCsuch that, forh
## Ξ
sufficiently small,
## ‖bχ
ξ
χ
η
## ‖
## H
m
≤Ch
d
## 2
## −m
## Ξ
## (4.1.6)
and
## ∥
## ∥
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## H
m−1
≤Ch
d
## 2
## −m−1
## Ξ
## .(4.1.7)
Proof.By Lemma 2.7.5,
## ‖bχ
ξ
χ
η
## ‖
## H
m
## ≤C
## (
## ‖b‖
## H
m
## ‖χ
ξ
χ
η
## ‖
## L
## ∞
## +‖b‖
## L
## ∞
## ‖χ
ξ
χ
η
## ‖
## H
m
## )
## .
We know‖b‖
## L
## ∞
## ≤ ‖b‖
## H
m
=C,  and‖χ
ξ
χ
η
## ‖
## L
## ∞
## ≤ ‖χ
ξ
## ‖
## L
## ∞
## ‖χ
η
## ‖
## L
## ∞
## ≤C.
Another application of Lemma 2.7.5 then gives
## ‖bχ
ξ
χ
η
## ‖
## H
m
## ≤C
## (
## 1 +
## (
## ‖χ
ξ
## ‖
## H
m
## ‖χ
η
## ‖
## L
## ∞
## +‖χ
ξ
## ‖
## L
## ∞
## ‖χ
η
## ‖
## H
m
## ))
## .
## Again‖χ
ξ
## ‖
## L
## ∞
≤C, and our bump estimate gives‖χ
ξ
## ‖
## H
m
≤Cq
d
## 2
## −m
ξ
## .  This
proves (4.1.6).
The proof of (4.1.7) is somewhat more involved.  We start by covering
Mwith finitely many coordinate patches Ω
## 1
## ,...,Ω
## K
, where
## Ω
k
## ⊆b(q
k
## ,r
## M
## /3)
for  eachkandq
k
∈M.   HereKis  a  fixed  constant  guaranteed  by  the
compactness ofM.  We can use normal coordinates Exp
k
## = Exp
q
k
for each
k, since Exp
k
## :U
k
## →Ω
k
is bijective (hereU
k
## = Exp
## −1
k
## (Ω
k
## )).  Letτ
## 1
## ,...,τ
## K
## 56

be a partition of unity subordinate to Ω
## 1
## ,...,Ω
## K
## .  Then
## ∥
## ∥
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∥
## ∥
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## 2
## W
m−1
## 2
## ≤C
## K
## ∑
k=1
## ∥
## ∥
τ
k
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## 2
## W
m−1
## 2
## (Ω
k
## )
## .
## (4.1.8)
Using the metric equivalence from Lemma 2.6.1,
## ∥
## ∥
τ
k
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## W
m−1
## 2
## (Ω
k
## )
## ≤C
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
τ
k
## ∑
i,j
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## W
m−1
## 2
## (U
k
## )
## ,(4.1.9)
where we have abused notation slightly by writing, for example,a
ij
instead
ofa
ij
◦Exp
## −1
k
.  Thus by the triangle inequality,
## ∥
## ∥
τ
k
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## W
m−1
## 2
## (Ω
k
## )
## ≤
## ∑
i,j
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## W
m−1
## 2
## (U
k
## )
## .(4.1.10)
Using the metric equivalence from Lemma 2.6.1 again,
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## W
m−1
## 2
## (U
k
## )
## ≤C
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## H
m−1
## ,(4.1.11)
where byτ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
we mean the extension by zero of this function from
## Ω
k
to all ofM.  This is justified because the closure of the support ofτ
k
is contained in Ω
k
.  Letσbe a smooth cutoff function withσ= 1 on the
support ofτ
k
andσ= 0 onM\Ω
k
.  We now apply Lemma 2.7.5 twice to
## 57

obtain
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## H
m−1
## ≤C
## ((
## ∥
## ∥
τ
k
a
ij
## ∥
## ∥
## H
m−1
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## ∞
## +
## ∥
## ∥
τ
k
a
ij
## ∥
## ∥
## L
## ∞
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## H
m−1
## )
## ∥
## ∥
## ∥
## ∥
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## L
## ∞
## +
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
σ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## ∞
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
η
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## H
m−1
## )
## .
## (4.1.12)
TheH
m−1
andL
## ∞
norms ofτ
k
a
ij
can be absorbed into the inner constant,
since they depend only onaand the choice of partition of unity.  Similarly,
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## ∞
## ≤
## ∥
## ∥
τ
k
a
ij
## ∥
## ∥
## L
## ∞
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## ∞
≤Ch
## −1
## Ξ
by Lemma 3.2.7, since
## ∥
## ∥
## ∥
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## L
## ∞
## ≤ ‖|∇χ
ξ
## |‖
## L
## ∞
.  (Because
## ∣
## ∣
## ∣
## ∂χ
ξ
## ∂x
i
## ∣
## ∣
## ∣
is just one
of  the  directional  derivatives  that|∇χ
ξ
|is  the  maximum  of.)   The  last
norms in (4.1.8) to handle are theH
m−1
norms of the partial derivatives
of the Lagrange functions.
This  will  require  yet  another  pass  through  Euclidean  space  using  the
metric equivalence from Lemma 2.6.1 and the norm equivalence from Lemma
2.7.3.  First, the norm equivalence gives
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## W
m−1
## 2
## (M)
## =C
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## W
m−1
## 2
## (Ω
k
## )
## .
## 58

The metric equivalence now gives
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## W
m−1
## 2
## (U
k
## )
## ,
where  we  have  again  abused  notation  slightly  by  writing
## ∂χ
ξ
## ∂x
i
instead  of
## ∂χ
ξ
## ∂x
i
◦Exp
## −1
k
on  the  left  andσinstead  ofσ◦Exp
k
.   By  the  definition  of
## W
m−1
## 2
## (U
k
) and Leibniz’s rule,
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∑
## |α|≤m−1
## ∫
## U
k
## ∣
## ∣
## ∣
## ∣
## D
α
## (
σ(x)
## ∂χ
ξ
## ∂x
i
## (x)
## )
## ∣
## ∣
## ∣
## ∣
## 2
dx
## =C
## ∑
## |α|≤m−1
## ∫
## U
k
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ∑
β≤α
## D
β
σ(x)D
α−β
## ∂χ
ξ
## ∂x
i
## (x)
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## 2
dx.
Now, all the derivatives ofσare bounded, as is the number of times each
γwith|γ|≤m−1 appears.  Absorbing these bounds into the constant, we
have
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∑
## |γ|≤m−1
## ∫
## U
k
## ∣
## ∣
## ∣
## ∣
## D
γ
## ∂χ
ξ
## ∂x
i
## ∣
## ∣
## ∣
## ∣
## 2
dx.
For each suchγ,D
γ
## ∂χ
ξ
## ∂x
i
## =D
δ
χ
ξ
for someδwith|δ|≤m.  Hence
## ∥
## ∥
## ∥
## ∥
σ
## ∂χ
ξ
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## 2
## H
m−1
## ≤C
## ∑
## |δ|≤m
## ∫
## U
k
## ∣
## ∣
## D
δ
χ
ξ
## ∣
## ∣
## 2
dx=C‖χ
ξ
## ‖
## 2
## W
m
## 2
## (U
k
## )
≤C‖χ
ξ
## ‖
## 2
## W
m
## 2
## (M)
≤C‖χ
ξ
## ‖
## 2
## H
m
≤Cq
d−2m
## Ξ
## ,
## (4.1.13)
where we have used the norm equivalence in the second-to-last inequality
and our bump estimate in the last.  The result now follows by successively
plugging back in to (4.1.12), (4.1.11), (4.1.10), (4.1.9), and (4.1.8).
Lemma 4.1.14.SupposeΦ
m
provides an energy estimate with parameter
m >
d
## 2
+ 1, and letΞandΛbe finite sets of centers(bothΠ
## L
-unisolvent if
## 59

## Φ
m
is conditionally positive definite with respect toΠ
## L
).  Assume#Ξ<#Λ
andh
## Λ
< h
## Ξ
.   LetB
## Ξ
andB
## Λ
## Ξ
be  the  stiffness  matrix  and  quadratized
stiffness matrix, respectively, whereΞis the set of centers use for Galerkin
approximation  andΛis  the  set  of  centers  used  for  quadrature.  There  is  a
constantCsuch that, forh
## Ξ
sufficiently small, the uniform estimate
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## ≤C
## (
h
## Λ
h
## Ξ
## )
m
h
d
## 2
## −1
## Ξ
h
## −1
## Λ
## ,
holds for allξ,η∈Ξ.
Proof.Our quadrature rule, Theorem 3.1.1, along with Theorem 4.1.5 gives
## ∣
## ∣
## ∣
## ∣
## ∫
## M
bχ
ξ
χ
η
dμ−Q
## Λ
## (bχ
ξ
χ
η
## )
## ∣
## ∣
## ∣
## ∣
≤Ch
m
## Λ
## ‖bχ
ξ
χ
η
## ‖
## H
m
≤Ch
m
## Λ
h
d
## 2
## −m
## Ξ
## .
## Similarly,
## ∣
## ∣
## ∣
## ∣
## ∫
## M
a
## ]
## (∇χ
ξ
## ,∇χ
η
)dμ−Q
## Λ
## (
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## )
## ∣
## ∣
## ∣
## ∣
≤Ch
m−1
## Λ
## ∥
## ∥
a
## ]
## (∇χ
ξ
## ,∇χ
η
## )
## ∥
## ∥
## H
m−1
≤Ch
m−1
## Λ
h
d
## 2
## −m−1
## Ξ
## ,
and the result follows by the triangle inequality.
Corollary 4.1.15.Adopt the notation and assumptions from Lemma 4.1.14.
There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
## ≤C
## (
h
## Λ
h
## Ξ
## )
m
h
## −
d
## 2
## −1
## Ξ
h
## −1
## Λ
## .
Proof.We have
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## ∣
.  By the lemma,  for
## 60

fixedξ∈Ξ we have
## ∑
η∈Ξ
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## ≤(#Ξ)·C
## (
h
## Λ
h
## Ξ
## )
m
h
d
## 2
## −1
## Ξ
h
## −1
## Λ
## ,
and #Ξ∼h
## −d
## Ξ
yields the result.
The uniform estimate from Lemma 4.1.14 allows us to prove Corollary
4.1.15; however, we are not taking advantage of our energy estimates.  To
do so, we fixξ∈Ξ and break the sum in the proof into an “inner sum”
and an “outer sum”.  The uniform estimate will be used on the inner sum,
whereas the bounds on the entries of the stiffness and quadratized stiffness
matrices from Lemmas 3.4.1 and 4.1.1 will be used on the outer sum.
Lemma 4.1.16.Adopt the notation and assumptions from Lemma 4.1.14.
SupposeΦprovides an algebraic energy estimate as in (3.2.3).  Assume in
addition that
h
## Λ
≥diam(M)
## −2−
## 2
m−1
h
## 3−
d−8
## 2m−2
## Ξ
## .(4.1.17)
There is a positive constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−
d
## 2
## −1
h
d
## 2
## −2
## Ξ
## .
Proof.We  want  to  bound
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
=  max
ξ∈Ξ
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## ∣
## .   Fixing
ξ∈Ξ andr
## 0
>0, we split the sum into an inner sum and an outer sum:
## ∑
η∈Ξ
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## =
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## +
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
## .
Let us consider the outer sum first.  DivideMinto annuliA
n
with outer
radiusnh
## Ξ
and inner radius (n−1)h
## Ξ
## ,n= 1,...,n
max
, and letn
## 0
## =b
r
## 0
h
## Ξ
c.
## 61

By Lemmas 3.4.1 and 4.1.1,
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
## ∑
η∈Ξ∩A
n
## (
## 1 +
dist(ξ,η)
h
## Ξ
## )
## −2m
≤Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
## (#Ξ∩A
n
## )
## (
## 1 +
## (n−1)h
## Ξ
h
## Ξ
## )
## −2m
=Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
## (#Ξ∩A
n
## )n
## −2m
## .
Now, Vol(A
n
## )∼n
d−1
h
d
## Ξ
, and hence #Ξ∩A
n
## ∼
n
d−1
h
d
## Ξ
q
d
## Ξ
## =ρ
d
n
d−1
## .  There-
fore
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
n
d−2m−1
≤Ch
d−2
## Ξ
## ∞
## ∑
n=n
## 0
n
d−2m−1
## .
An integral comparison gives
## ∞
## ∑
n=n
## 0
n
d−2m−1
## ≤n
d−2m−1
## 0
## (
## 1 +
n
## 0
## 2m−d
## )
## ≤h
## 2m−d+1
## Ξ
r
d−2m−1
## 0
## (
## 1 +
r
## 0
## 2m−d
## )
## .
Hence we obtain
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
## 2m−1
## Ξ
r
d−2m−1
## 0
## (
## 1 +
r
## 0
## 2m−d
## )
## .(4.1.18)
Let  us  now  examine  the  inner  sum.   The  set  of  centers  Ξ∩b(ξ,r
## 0
)  has
cardinality bounded by
## Vol (b(ξ,r
## 0
## ))
## Vol (b(ξ,q
## Ξ
## ))
## ∼
r
d
## 0
q
d
## Ξ
## =ρ
d
r
d
## 0
h
## −d
## Ξ
## .
## 62

The uniform estimate from Lemma 4.1.14, then, gives
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
## −
d
## 2
## −m−1
## Ξ
h
m−1
## Λ
r
d
## 0
## .(4.1.19)
Assume for now thatr
## 0
≤diam(M).  Then (4.1.18) becomes
## ∑
η∈b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
## 2m−1
## Ξ
r
d−2m−1
## 0
## (
## 1 +
diam(M)
## 2m−d
## )
=Ch
## 2m−1
## Ξ
r
d−2m−1
## 0
## .
## (4.1.20)
Setting the results from (4.1.19) and (4.1.20) equal and solving forr
## 0
gives
r
## 0
## =h
## 6m+d
## 4m+2
## Ξ
h
## −
m−1
## 2m+1
## Λ
## =h
## 3
## 2
## +
d−3
## 4m+2
## Ξ
h
## −
## 1
## 2
## +
## 3
## 4m+2
## Λ
## .
The condition onh
## Λ
, (4.1.17), ensuresr
## 0
≤diam(M).  Therefore (4.1.20)
holds for thisr
## 0
, and plugging it into either (4.1.19) or (4.1.20) yields the
result.
Lemma 4.1.21.Adopt the notation and assumptions from Lemma 4.1.14.
SupposeΦprovides an  exponential energy  estimate as  in (3.2.4).  Assume
in addition that
ν
## 1
m−1
h
## 1+
d−4
## 2m−2
## Ξ
e
−diam(M)/νh
## Ξ
## (m−1)
## ≤h
## Λ
## <
## (
ν
e
## )
h
## 1+
d−4
## 2m−2
## Ξ
## .(4.1.22)
There is a positive constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−1
h
d
## 2
## −2
## Ξ
h
## −
## Λ
## ,
where=
dlog|log(h
## Λ
## )|
## |log(h
## Λ
## )|
## .
Proof.We again have
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## B
## Λ
ξη
## −B
## Λ
ξη
## ∣
## ∣
## ∣
, and for a fixed
## 63

ξ∈Ξ we split the sum into an inner sum and an outer sum as in the proof
of Lemma 4.1.16.  Handling the outer sum first, we have by Lemmas 3.4.1
and 4.1.1 that
## ∑
η∈Ξ∩b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
## ∑
η∈Ξ∩A
n
exp
## (
## −ν
dist(ξ,η)
h
## Ξ
## )
## .
≤Ch
d−2
## Ξ
n
max
## ∑
n=n
## 0
## (#(Ξ∩A
n
## ))e
## −ν(n−1)
≤Ch
d−2
## Ξ
## ∞
## ∑
n=n
## 0
n
d−1
e
## −ν(n−1)
## .
An integral comparison now gives
## ∞
## ∑
n=n
## 0
n
d−1
e
## −ν(n−1)
## ≤e
## −ν(n
## 0
## −1)
## (
n
d−1
## 0
## +
d−1
## ∑
i=0
## (−1)
d−i
## (d−1)!
i!(−ν)
d−i
n
i
## 0
## )
≤C(d−1)n
d−1
## 0
e
## −νn
## 0
## ≤C
## ′
h
## 1−d
## Ξ
r
d−1
## 0
e
## −νr
## 0
## /h
## Ξ
## .
where we have taken
## C=e
ν
max
## {
## 1,max
## {
## ∣
## ∣
## ∣
## ∣
## (d−1)!
i!(−ν)
d−i
## ∣
## ∣
## ∣
## ∣
## :i∈{0,...,d−1}
## }}
## .
We thus arrive at
## ∑
η∈b(ξ,r
## 0
## )
c
## ∣
## ∣
## B
ξη
## −B
## Λ
ξη
## ∣
## ∣
≤Ch
## −1
## Ξ
r
d−1
## 0
e
## −νr
## 0
## /h
## Ξ
## .(4.1.23)
The bound in (4.1.19) remains the same. Our task now is to find the value of
r
## 0
that makes (4.1.19) and (4.1.23) equal.  This requires solving an equation
of the formar
## 0
## =e
br
## 0
, wherea=h
## −
d
## 2
## −m
## Ξ
h
m−1
## Λ
andb=−νh
## −1
## Ξ
## .  Since
b
a
## =
## −νh
m+
d
## 2
## −1
## Ξ
h
## 1−m
## Λ
<0, there is a unique solution; namelyr
## 0
## =−
## 1
b
## W
## (
## −
b
a
## )
## ,
## 64

whereWis the principal branch of the Lambert W function.  Hence,
r
## 0
## =ν
## −1
h
## Ξ
## W
## (
νh
m+
d
## 2
## −1
## Ξ
h
## 1−m
## Λ
## )
## .(4.1.24)
The lower bound onh
## Λ
in (4.1.22) ensures thatr
## 0
≤diam(M).  Plugging
(4.1.24) into either (4.1.19) or (4.1.23) yields
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
≤Ch
d
## 2
## −m−1
## Ξ
h
m−1
## Λ
## W
## (
νh
m+
d
## 2
## −1
## Ξ
h
## 1−m
## Λ
## )
d
## .
The upper bound onh
## Λ
in (4.1.22) ensuresνh
m+
d
## 2
## −1
## Ξ
h
## 1−m
## Λ
> e, and by [19,
Theorem 2.1]W(t)<log(t) fort > e.  Hence the result now follows from
## ∣
## ∣
## ∣
log
## (
νh
m+
d
## 2
## −1
## Ξ
h
## 1−m
## Λ
## )
## ∣
## ∣
## ∣
## =
## ∣
## ∣
## ∣
## ∣
log(ν) +
## (
m+
d
## 2
## −1
## )
log(h
## Ξ
## )−(m−1)log(h
## Λ
## )
## ∣
## ∣
## ∣
## ∣
≤C|log(h
## Λ
## )|
Lemma 4.1.25.Adopt the notation and assumptions from Lemma 4.1.14.
LetC
## 0
= max{C
## 1
## ,C
## 2
## ,C
## 3
,1}, whereC
## 1
## ,C
## 2
, andC
## 3
are the constants from
Lemmas 3.4.4, 4.1.16, and 4.1.21, respectively.  Assumeh
## Λ
is small enough
that
h
## Λ
## ≤C
## 4
## 2+d−2m
## 0
h
## 1+
## 4d+4
## 2m−d−2
## Ξ
## 2
## 2
## 2+d−2m
## .
There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
≤Ch
## −d
## Ξ
## .
Remark4.1.26.Before we prove the Lemma, we explain the upper bound on
h
## Λ
.  In [22], the authors make a the assumption that the quadrature weights
## 65

## {w
ζ
## }
ζ∈Λ
## ({w
y
## }
y∈Y
in their notation) are not only positive,  but bounded
below byCh
d
## Λ
(Ch
## 2
## Y
in their notation) - see [22, Assumption 4.15].  This is
a conjecture for quadrature on quasi-uniform sets of centers supported by
some amount of empirical data, but has yet to be proven, even onS
## 2
## .  If
that assumption is made, then the result in Lemma 4.1.25 can be achieved
without placing the restrictive upper bound onh
## Λ
- the proof is analogous
to  the  proof  found  in  [22,  Theorem  7.7].   We  will  revisit  these  ideas  in
Remark 4.2.9, and give the analogous proof in Lemma 4.2.12.
Remark4.1.27.We note that the 2 in the upper bound forh
## Λ
was chosen
for simplicity.  It can be replaced by 1 +for any >0.  The idea is to
ensure the convergence of the Neumann series that arises in the proof.
Proof.Note that
## B
## Λ
## Ξ
## =B
## Ξ
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
## ,
and hence
## (
## B
## Λ
## Ξ
## )
## −1
## =
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
## −1
## B
## −1
## Ξ
## .
The  matricesB
## −1
## Ξ
andB
## Ξ
## −B
## Λ
## Ξ
are  self-adjoint,  and  hence
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 1
and
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
.  By Lemmas 3.4.4 and 4.1.16,
then,
## ∥
## ∥
## B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## )
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 2
## ≤C
## 2
## 0
h
## −m−
## 3d
## 2
## −1
## Ξ
h
m−
d
## 2
## −1
## Λ
## .
The condition onh
## Λ
ensures that
## ∥
## ∥
## B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## )
## ∥
## ∥
## 2
## ≤
## 1
## 2
## ,
## 66

and this in turn ensures the convergence of the Neumann series
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
## −1
## =
## ∞
## ∑
n=0
## (
## B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
n
## .
## Therefore
## ∥
## ∥
## ∥
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## 2
## ≤
## ∞
## ∑
n=0
## (
## 1
## 2
## )
n
## = 2,
and hence
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## ∥
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −B
## Λ
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
≤Ch
## −d
## Ξ
by another application of Lemma 3.4.4, since
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 1
as well.
## 4.2    Error Estimates
We  are  now  ready  to  prove  error  estimates  for  our  quadratized  Galerkin
approximation toLu=f.  To recapitulate, the Galerkin approximation is
u
## Ξ
## =
## ∑
ξ∈Ξ
γ
ξ
χ
ξ
## ,
whose coefficientsγ={γ
ξ
## }
ξ∈Ξ
are obtained by solving the linear system
## B
## Ξ
γ=ω, whereB
## Ξ
is the stiffness matrix andω={ω
ξ
## }
ξ∈Ξ
has entries
ω
ξ
## =
## ∫
## M
χ
ξ
f dμ.
The quadratized Galerkin approximation,
u
## Λ
## Ξ
## =
## ∑
ξ∈Ξ
γ
## Λ
ξ
χ
ξ
## ,
## 67

has coefficientsγ
## Λ
## =
## {
γ
## Λ
ξ
## }
ξ∈Ξ
obtained by solving the systemB
## Λ
## Ξ
γ
## Λ
## =ω
## Λ
## ,
whereω
## Λ
## =
## {
ω
## Λ
ξ
## }
ξ∈Ξ
is the vector with entriesω
## Λ
ξ
## =Q
## Λ
## (χ
ξ
f).
In Theorem 4.2.1, we estimate the error between the Galerkin approxi-
mation and the quadratized Galerkin approximation.  Corollary 4.2.7 then
gives the full error estimate between the weak solution toLu=fand the
quadratized Galerkin approximation.  It is simply the result of an applica-
tion of the triangle inequality.
Theorem 4.2.1.SupposeΦ
m
provides an energy estimate with parameter
m >
d
## 2
+ 1,  and  letΞandΛbe  finite  subsets  ofM(bothΠ-unisolvent  if
## Φ
m
is conditionally positive definite with respect toΠ).  Assume#Ξ<#Λ,
and  thath
## Λ
satisfies  the  upper  bound  from  Lemma  4.1.25.   Letf∈H
s
## ,
wheres=m−1if
d
## 2
+ 1< m≤
d
## 2
## + 2ands=m−2ifm >
d
## 2
## + 2.  Letu
## Λ
## Ξ
be the quadratized Galerkin approximation toLu=f, usingΞfor Galerkin
approximation andΛfor quadrature.
SupposeΦ
m
provides an algebraic energy estimate as in (3.2.3).  Assume
h
## Λ
also satisfies the bound from Lemma 4.1.16.  There is a positive constant
Csuch that
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−
d
## 2
## −1
h
## −
d
## 2
## −2
## Ξ
## ‖f‖
## H
s
## .
Now supposeΦ
m
provides an exponential energy estimate as in (3.2.4).
## Assumeh
## Λ
also satisfies the bounds from Lemma 4.1.21.  There is a positive
constantCsuch that
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−1
h
## −
d
## 2
## −2
## Ξ
h
## −
## Λ
## ‖f‖
## H
s
## ,
where=
dlog|log(h
## Λ
## )|
## |log(h
## Λ
## )|
## .
Remark4.2.2.Before we prove the result, we explain why there are different
## 68

cases  fors.   If
d
## 2
+ 1< m≤
d
## 2
+ 2,  then  we  have  a  problem  if  we  take
f∈H
m−2
,  becausem−2≤
d
## 2
,  and  the  Sobolev  embedding  theorem
doesn’t guaranteeH
m−2
⊂C(M).  Hence, we don’t know thatf∈H
m−2
is
a continuous function, or even that it is defined everywhere.  In that case we
don’t know that we can samplefon Λ, which is necessary for quadrature.
That is why we insist onf∈H
m−1
in that case, and notH
m−2
## .
Proof.By theL
## 2
-stability of the Lagrange basis,
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## =
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
## (
γ
ξ
## −γ
## Λ
ξ
## )
χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
d/2
## Ξ
## ∥
## ∥
γ−γ
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
=Ch
d/2
## Ξ
## ∥
## ∥
## ∥
## B
## −1
## Ξ
ω−
## (
## B
## Λ
## Ξ
## )
## −1
ω
## Λ
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## .
Adding and subtracting
## (
## B
## Λ
## Ξ
## )
## −1
ωinside the norm and applying the triangle
inequality gives
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
≤Ch
d/2
## Ξ
## (
## ∥
## ∥
## ∥
## (
## B
## −1
## Ξ
## −
## (
## B
## Λ
## Ξ
## )
## −1
## )
ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## (
ω−ω
## Λ
## )
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## .
Noting thatB
## −1
## Ξ
## −
## (
## B
## Λ
## Ξ
## )
## −1
## =
## (
## B
## Λ
## Ξ
## )
## −1
## (
## B
## Λ
## Ξ
## −B
## Ξ
## )
## B
## −1
## Ξ
, we have
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
≤Ch
d/2
## Ξ
## (
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## B
## Λ
## Ξ
## −B
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
ω−ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## .
## 69

Factoring out
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
and applying Lemma 4.1.25 gives
## ∥
## ∥
u
## Ξ
## −u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
≤Ch
## −d/2
## Ξ
## (
## ∥
## ∥
## B
## Λ
## Ξ
## −B
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
ω−ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## .
## (4.2.3)
We bound the quantities in parentheses in (4.2.3) separately.  For the first,
we begin by using theL
## 2
stability of the Lagrange basis again to obtain
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## =‖γ‖
## `
## 2
## (Ξ)
≤Ch
## −d/2
## Ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
γ
ξ
χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
=Ch
## −d/2
## Ξ
## ‖u
## Ξ
## ‖
## L
## 2
≤Ch
## −d/2
## Ξ
## (
## ‖u−u
## Ξ
## ‖
## L
## 2
## +‖u‖
## L
## 2
## )
## ,
whereuis the weak solution toLu=f.  By regularity‖u‖
## L
## 2
## ≤ ‖u‖
## H
## 2
## ≤
## C‖f‖
## L
## 2
≤C‖f‖
## H
s
, and by Theorem 3.3.6,‖u−u
## Ξ
## ‖
## L
## 2
≤Ch
m−1
## Ξ
## ‖f‖
## H
m−2
## ≤
## Ch
m−1
## Ξ
## ‖f‖
## H
s
## .  Hence,
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
≤Ch
## −d/2
## Ξ
## (
h
m−1
## Ξ
## + 1
## )
## ‖f‖
## H
s
≤Ch
## −d/2
## Ξ
## ‖f‖
## H
s
## .
In the case of an algebraic energy estimate, we apply Lemma 4.1.16, keeping
in mind that
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## Ξ
## −B
## Λ
## Ξ
## ∥
## ∥
## 1
, to obtain
## ∥
## ∥
## B
## Λ
## Ξ
## −B
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−
d
## 2
## −1
h
## −2
## Ξ
## ‖f‖
## H
s
## .(4.2.4)
In the case of an exponential energy estimate, Lemma 4.1.21 gives
## ∥
## ∥
## B
## Λ
## Ξ
## −B
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## ≤C
## (
h
## Λ
h
## Ξ
## )
m−1
h
## −2
## Ξ
h
## −
## Λ
## ‖f‖
## H
s
## .(4.2.5)
For the second quantity in the parentheses in (4.2.3),  we begin with our
## 70

quadrature rule and Lemma 2.7.5:
## ∣
## ∣
ω
ξ
## −ω
## Λ
ξ
## ∣
## ∣
≤Ch
s
## Λ
## (
## ‖f‖
## H
s
## ‖χ
ξ
## ‖
## L
## ∞
## +‖f‖
## L
## ∞
## ‖χ
ξ
## ‖
## H
s
## )
## .
## Now,‖f‖
## L
## ∞
≤C‖f‖
## H
s
## ,‖χ
ξ
## ‖
## L
## ∞
≤C, and the Bernstein inequality from
[14, Theorem 3] gives‖χ
ξ
## ‖
## H
s
≤Ch
d
## 2
## −s
## Ξ
## .  Thus,
## ∣
## ∣
ω
ξ
## −ω
## Λ
ξ
## ∣
## ∣
≤Ch
s
## Λ
## (
## 1 +h
d
## 2
## −s
## Ξ
## )
## ‖f‖
## H
s
## ≤C
## (
h
## Λ
h
## Ξ
## )
s
h
d/2
## Ξ
## .
Taking the`
## 2
(Ξ)-norm of these,
## ∥
## ∥
ω−ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
## ≤(#Ξ)
## 1/2
## ·C
## (
h
## Λ
h
## Ξ
## )
s
h
d/2
## Ξ
## ‖f‖
## H
s
## ≤C
## (
h
## Λ
h
## Ξ
## )
s
## ‖f‖
## H
s
## .
## (4.2.6)
In the algebraic case, (4.2.6) is clearly controlled by (4.2.4) fors=m−1
ands=m−2.  Ifs=m−1, then (4.2.6) is clearly controlled by (4.2.5)
as well.  For the cases=m−2, we observe thath
## Λ
## /h
## Ξ
## ≤h
## −2
## Ξ
h
## −
## Λ
, since
surelyh
## 1+
## Λ
## ≤h
## −1
## Ξ
.   Hence  (4.2.6)  is  controlled  by  (4.2.5)  in  this  case  as
well.  The results now follow by putting (4.2.6) along with either (4.2.4) or
(4.2.5) into (4.2.3).
Corollary 4.2.7.Adopt the notation and assumptions from Theorem 4.2.1.
SupposeΦprovides an algebraic energy estimate, and assume the bound on
h
## Λ
from Lemma 4.1.16 holds.  There is a positive constantCsuch that, for
h
## Ξ
sufficiently small,
## ∥
## ∥
u−u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +
## (
h
## Λ
h
## Ξ
## )
m−
d
## 2
## −1
h
## −
d
## 2
## −2
## Ξ
## )
## ‖f‖
## H
s
## .
Now  supposeΦprovides  an  exponential  energy  estimate,  and  assume
## 71

the bounds onh
## Λ
from Lemma 4.1.21 hold.  There is a positive constantC
such that, forh
## Ξ
sufficiently small,
## ∥
## ∥
u−u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +
## (
h
## Λ
h
## Ξ
## )
m−1
h
## −
d
## 2
## −2
## Ξ
h
## −
## Λ
## )
## ‖f‖
## H
s
## ,
where=
dlog|log(h
## Λ
## )|
## |log(h
## Λ
## )|
## .
Remark4.2.8.We come now to one of our main points: we can get the same
result with quadrature as we can without by choosingh
## Λ
appropriately.  Let
us put aside the upper bound onh
## Λ
from Lemma 4.1.25 for a moment.  We
take as an ansatz thath
## Λ
## =h
p
## Ξ
, and callptheoversampling exponent.  The
optimal  oversampling  exponentp
## ∗
is the smallest value ofpthat ensures
that the quadrature error is on par with the Galerkin error.  It is obtained
by equating the quantities in the parentheses in Corollary 4.2.7 and solving
forh
## Λ
.  In the case of an algebraic energy estimate,
p
## ∗
## = 2 +
## 2d+ 4
## 2m−d−2
## .
For an exponential energy estimate,
p
## ∗
## = 2 +
d+ 4 + 4
## 2m−2−2
## .
Remark4.2.9.Let us examine again the restriction onh
## Λ
in Lemma 4.1.25:
h
## Λ
## ≤C
## 1
h
## 1+
## 4d+4
## 2m−d−2
## Ξ
, with
## C
## 1
## =C
## 4
## 2+d−2m
## 0
## 2
## 2
## 2+d−2m
## .
We know thatC
## 0
depends only on Ξ via the mesh ratio, but we don’t know
what  it  is  exactly.   It  is  possible  that  this  restriction  is  already  stronger
than the one imposed by the optimal oversampling exponent.  If, however,
## 72

h
## Λ
is less thanC
## 1
(the lemma does say “forh
## Λ
sufficiently small”,  after
all) then the restriction becomesh
## Λ
## ≤h
p
## ∗
## Ξ
, with thebaseline oversampling
exponent
p
## ∗
## = 1 +
## 4d+ 4
## 2m−d−2
## .
Noting that in this casep
## ∗
## ≤p
## ∗
, we now get a range for the oversampling
exponent;  namely,p∈[p
## ∗
## ,p
## ∗
].  Ifp < p
## ∗
, then Lemma 4.1.25, and hence
Corollary 4.2.7, doesn’t hold. Takingp > p
## ∗
doesn’t fundamentally improve
the error estimate, since the Galerkin error stays the same.  We note that
withpin  the  range  [p
## ∗
## ,p
## ∗
],  all  of  the  bounds  onh
## Λ
in  Lemmas  4.1.16,
4.1.21, and 4.1.25 hold ifh
## Λ
## =h
p
## Ξ
, providedh
## Ξ
is sufficiently small.
Remark4.2.10.With an additional assumption we can relax the bound on
h
## Λ
in Lemma 4.1.25.  Specifically, if we know that the quadrature weights
are  positive,  then  we  can  apply  Lemma  4.2.12.   It  has  been  conjectured
that for quasi-uniform sets of centers Λ, the quadrature weights are posi-
tive; however, this has yet to be proved.  There is, however, some empirical
evidence to support the conjecture, which we exhibit in Figure 4.2.10.  The
data comes from experiments onSO(3) with kernels Φ
m
that will be intro-
duced in Chapter 5.
## 73

Figure 4.1:  Plots of log(w
min
) versus log(q
## Λ
), wherew
min
is the minimum
quadrature weight for centers Λ, for samples Λ ofSO(3) of sizes 396, 896,
1005, 1872, and 3749.
As mentioned in Remark 4.1.26, we don’t need a baseline oversampling
exponent if we know the quadrature weights are positive.  In that case we
have a more direct way to get the result from Lemma 4.1.25.  We need the
following technical lemma for the proof.
Lemma 4.2.11.Adopt the notation and assumptions from Lemma 4.1.14,
supposeu∈V
## Ξ
,  and  letI
## Λ
be  the  interpolation  operator  relative  to  the
centersΛ.  There is a constantCsuch that, providedh
## Λ
≤Ch
## Ξ
## ,
## 1
## 2
## ‖u‖
## L
## 2
## ≤‖I
## Λ
u‖
## L
## 2
## ≤
## 3
## 2
## ‖u‖
## L
## 2
## .
## 74

Proof.Two applications of the triangle inequality give
## ‖u‖
## L
## 2
−‖u−I
## Λ
u‖
## L
## 2
## ≤‖I
## Λ
u‖
## L
## 2
## ≤‖u‖
## L
## 2
+‖u−I
## Λ
u‖
## L
## 2
## .
Thus, it suffices to show aCexists that ensures‖u−I
## Λ
u‖
## L
## 2
## ≤
## 1
## 2
## ‖u‖
## L
## 2
## .
By our interpolation error estimate,
‖u−I
## Λ
u‖
## L
## 2
## ≤C
## 1
h
m
## Λ
## ‖u‖
## H
m
## .
## By  [14,  Theorem  10],‖u‖
## H
m
## ≤C
## 2
h
## −m
## Ξ
foru∈V
## Ξ
.   Applying  it  to  our
situation gives
‖u−I
## Λ
u‖
## L
## 2
## ≤C
## 1
## C
## 2
## (
h
## Λ
h
## Ξ
## )
m
## ‖u‖
## L
## 2
## .
Hence, it suffices to takeC= (2C
## 1
## C
## 2
## )
## −1/m
## .
Lemma 4.2.12.Adopt the notation and assumptions from Lemma 4.1.14.
Assume the quadrature weightsw={w
ζ
## }
ζ∈Λ
are positive, and letw
min
be
the  minimum  quadrature  weight.   There  is  a  constantC≥h
## Λ
w
## −1
min
such
that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## (
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## 2
≤Ch
## −d
## Ξ
## .
Proof.As in the proof of Lemma 3.4.4, it suffices to show thatλ
min
## (
## B
## Λ
## Ξ
## )
## ≥
## Cq
d
## Ξ
,  and  for  that  it  suffices  to  show  thatv
## T
## B
## Λ
## Ξ
v≥Cq
d
## Ξ
## ‖v‖
## 2
## `
## 2
## (Ξ)
for  all
vectorsv={v
ξ
## }
ξ∈Ξ
.  Also as in the proof of Lemma 3.4.4, we will need the
## L
## 2
stability of the Lagrange basis{χ
ξ
## }
ξ∈Ξ
, but now we will also need theL
## 2
stability of the Lagrange basis{ ̃χ
ζ
## }
ζ∈Λ
; namely, that there are constants
## 75

## D
## 1
andD
## 2
such that
## D
## 1
q
d/2
## Λ
## ‖y‖
## `
## 2
## (Λ)
## ≤
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ζ∈Λ
y
ζ
## ̃χ
ζ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## ≤D
## 2
q
d/2
## Λ
## ‖y‖
## `
## 2
## (Λ)
for  all  vectorsy={y
ζ
## }
ζ∈Λ
.   So,  starting  with  a  vectorv={v
ξ
## }
ξ∈Ξ
,  let
u=
## ∑
ξ∈Ξ
v
ξ
χ
ξ
, so that∇u=
## ∑
ξ∈Ξ
v
ξ
## ∇χ
ξ
, and lety={y
ζ
## }
ζ∈Λ
## =u|
## Λ
## .
We have
v
## T
## B
## Λ
## Ξ
v=
## ∑
ξ∈Ξ
## ∑
η∈Ξ
v
ξ
v
η
## ∑
ζ∈Λ
## (
a
## ]
## (ζ) (∇χ
ξ
## (ζ),∇χ
η
## (ζ)) +b(ζ)χ
ξ
## (ζ)χ
η
## (ζ)
## )
w
ζ
## =
## ∑
ζ∈Λ
## 
## 
a
## ]
## (ζ)
## 
## 
## ∑
ξ∈Ξ
v
ξ
## ∇χ
ξ
## (ζ),
## ∑
η∈Ξ
v
η
## ∇χ
η
## (ζ)
## 
## 
## +b(ζ)
## 
## 
## ∑
ξ∈Ξ
v
ξ
χ
ξ
## (ζ)
## 
## 
## 
## 
## ∑
η∈Ξ
v
η
χ
η
## (ζ)
## 
## 
## 
## 
w
ζ
## =
## ∑
ζ∈Λ
## (
a
## ]
## (ζ) (∇u(ζ),∇u(ζ)) +b(ζ)u(ζ)
## 2
## )
w
ζ
## =Q
## Λ
## (
a
## ]
## (∇u,∇u) +bu
## 2
## )
## .
Now,  the  positive  definiteness  ofaensuresa
## ]
(∇u,∇u)>0.   Using  this,
the assumption thatw
ζ
≥Ch
d
## Λ
for eachζ∈Λ, and the assumption that
b≥b
## 0
>0 gives
v
## T
## B
## Λ
## Ξ
v≥Q
## Λ
## (
bu
## 2
## )
## =
## ∑
ζ∈Λ
b(ζ)u(ζ)
## 2
w
ζ
≥Cb
## 0
h
d
## Λ
## ∑
ζ∈Λ
u(ζ)
## 2
=Cb
## 0
h
d
## Λ
## ‖y‖
## 2
## `
## 2
## (Λ)
## .
Let ̃u=I
## Λ,L
u=
## ∑
ζ∈Λ
y
ζ
## ̃χ
ζ
be the interpolant touon Λ.  TheL
## 2
stability
## 76

of{ ̃χ
ζ
## }
ζ∈Λ
gives
v
## T
## B
## Λ
## Ξ
v≥Cb
## 0
h
d
## Λ
## D
## −1
## 2
q
## −d
## Λ
## ‖ ̃u‖
## 2
## L
## 2
=Cb
## 0
## D
## −1
## 2
ρ
d
## ‖ ̃u‖
## 2
## L
## 2
## .
## By Lemma 4.2.11,‖ ̃u‖
## 2
## L
## 2
## ≥
## 1
## 4
## ‖u‖
## 2
## L
## 2
, and this along with theL
## 2
stability of
## {χ
ξ
## }
ξ∈Ξ
gives
v
## T
## B
## Λ
## Ξ
v≥
## 1
## 4
## Cb
## 0
## D
## −1
## 2
ρ
d
## ‖u‖
## 2
## L
## 2
## ≥
## 1
## 4
## Cb
## 0
## D
## −1
## 2
ρ
d
## C
## 1
q
d
## Ξ
## ‖v‖
## 2
## `
## 2
## (Ξ)
## .
## 4.3    Algorithms
To implement our Galerkin approximation numerically, we need to calculate
the quadrature weights, the entries of the quadratized stiffness matrix, and
the Galerkin approximation. We postpone the algorithm for the quadrature
weights  to  Section  5.6,  when  we’ll  know  more  about  the  integrals  of  our
kernel and the functions in our auxiliary space.
## Suppose{b
## 1
## ,...,b
## N
}is a basis for our set of centers Ξ ={ξ
## 1
## ,...,ξ
## N
## }
for Galerkin approximation, let Λ ={ζ
## 1
## ,...,ζ
## M
}be our set of centers for
quadrature, and letw={w
## 1
## ,...,w
## M
}be the quadrature weights.  We wish
to form the quadratized stiffness matrixS= [S
ij
## ]
## N
i,j=1
, with entries
## S
i,j
## =Q
## Λ
## (
a
## ]
## (∇b
i
## ,∇b
j
## ) +bb
i
## ,b
j
## )
## =
## M
## ∑
k=1
w
k
## (
a
## ]
## (ζ
k
## ) (∇b
i
## ,∇b
j
## ) +b(ζ
k
## )b
i
## (ζ
k
## )b
j
## (ζ
k
## )
## )
## =
## M
## ∑
k=1
w
k
## (
d
## ∑
m,n=1
a
mn
## (ζ
k
## )
## ∂b
i
## ∂x
m
## (ζ
k
## )
## ∂b
j
## ∂x
n
## (ζ
k
## ) +b(ζ
k
## )b
i
## (ζ
k
## )b
j
## (ζ
k
## )
## )
## .
## 77

For  a  positive  definite  kernel  Φ,  one  can  simply  takeb
j
## =  Φ(·,ξ
j
## ).   The
following  algorithm  computes  the  quadratized  stiffness  matrix,  given  the
centers for quadrature, the quadrature weights, the coefficients in our dif-
ferential equation, and a basis forV
## Ξ
## .
Algorithm 1Quadratized Stiffness Matrix for〈u,v〉
a,b
## =λ
f
## (v).
## Input:  Λ  ={ζ
## 1
## ,...,ζ
## M
},  quadrature  weights{w
## 1
## ,...,w
## M
},  the  func-
tionsa
ij
andbfrom our differential equation,  and a basis{b
## 1
## ,...,b
## N
## }
forV
## Ξ
## .
Output:  The quadratized stiffness matrix.
•Form the basis matrices
## B=
## 
## 
## 
## 
## 
b
## 1
## (ζ
## 1
## )b
## 2
## (ζ
## 2
## )·b
## N
## (ζ
## 1
## )
b
## 1
## (ζ
## 2
## )b
## 2
## (ζ
## 2
## )···b
## N
## (ζ
## 2
## )
## .
## .
## .
## .
## .
## .
## .
## .
## .
b
## 1
## (ζ
## M
## )b
## 2
## (ζ
## M
## )···b
## N
## (ζ
## M
## )
## 
## 
## 
## 
## 
## ,
and
## B
i
## =
## 
## 
## 
## 
## 
## ∂
## ∂x
i
b
## 1
## (ζ
## 1
## )
## ∂
## ∂x
i
b
## 2
## (ζ
## 1
## )···
## ∂
## ∂x
i
b
## N
## (ζ
## 1
## )
## ∂
## ∂x
i
b
## 1
## (ζ
## 2
## )
## ∂
## ∂x
i
b
## 2
## (ζ
## 2
## )···
## ∂
## ∂x
i
b
## N
## (ζ
## 2
## )
## .
## .
## .
## .
## .
## .
## .
## .
## .
## ∂
## ∂x
i
b
## 1
## (ζ
## M
## )
## ∂
## ∂x
i
b
## 2
## (ζ
## M
## )···
## ∂
## ∂x
i
b
## N
## (ζ
## M
## )
## 
## 
## 
## 
## 
•ComputeD
bw
= diag(bw) andD
a
ij
w
= diag(a
ij
w).
•Obtain the stiffness matrixS=B
## ∗
## D
bw
## B+
d
## ∑
i,j=1
## B
## ∗
i
## D
a
ij
w
## B
j
## .
For  a  conditionally  positive  definite  kernel  Φ,  the  functions  Φ(·,ξ
j
## )
aren’t  even  inV
## Ξ
.   In  this  case  one  has  to  construct  a  basis  using  both
the  translates  of  the  kernel  and  the  functions  in  the  auxiliary  space  Π.
One can take the Lagrange basis,  or take the following approach,  due to
Wendland - see [29, Section 10.3] for details.  Choose a Π-unisolvent sub-
set  Ξ
## 0
## ={x
## 1
## ,...,x
## L
}of  Ξ,  set  Ξ
## 1
## =  Ξ\Ξ
## 0
## ={y
## 1
## ,...,y
## N−L
},  and  let
## {p
## 1
## ,...,p
## L
}be the Lagrange basis for Π.  Extend{p
## 1
## ,...,p
## L
}to a basis
## {b
## 1
## ,...,b
## N−L
## ,p
## 1
## ,...,p
## L
}by settingb
j
## = Φ(·,y
j
## )−
## ∑
## L
k=1
p
k
## (y
j
)Φ(·,x
k
## ).
## 78

Algorithm 2Basis forV
## Ξ
## .
Input:  kernel Φ, auxiliary space Π of dimensionQ, and a Π-unisolvent
set of centers Ξ ={ξ
## 1
## ,...,ξ
## N
## }.
Output:  a basis{b
## 1
## ,...,b
## N
}forV
## Ξ
## .
•Choose a (quasi-uniform) subset Ξ
## 0
## ={x
## 1
## ,...,x
## Q
## }.
•Let Ξ\Ξ
## 0
## = Ξ
## 1
## ={y
## 1
## ,...,y
## N−Q
## }.
•Form the Lagrange basis{p
## 1
## ,...,p
## Q
## }for Ξ
## 0
and Π.  (We can do this by
forming the collocation matrixK={φ
j
## (ξ
k
## )}
## Q
j,k=1
; then the coefficients
ofp
j
are in thej
th
column ofK
## −1
## .)
•For 1≤j≤N−Qlet
b
j
## = Φ(·,y
j
## )−
## Q
## ∑
k=1
p
k
## (y
j
)Φ(·,x
k
## ).
•ForN−Q+ 1≤j≤N, letb
j
## =p
j−N+Q
## .
•Then{b
## 1
## ,...,b
## N−Q
## ,p
## 1
## ,...,p
## Q
}is a basis forV
## Ξ
## .
Once the basis and stiffness matrix has been formed, given a function
fthat  is  to  be  the  right  side  of  our  differential  equation,  the  Galerkin
approximation  is  formed  by  pairing  the  basis  elements  with  coefficients
that are determined by a linear system.  That is described in Algorithm 3.
Algorithm 3Quadratized Galerkin Approximation to solution ofLu=f.
## Input:f|
## Λ
, the stiffness matrixS, the quadrature weightsw, and a basis
matrixBas formed in Algorithm 4.3..
Output:  the quadratized Galerkin approximation.
•ComputeD
bw
= diag(bw).
•Compute the RHSR=B
## ∗
## D
bw
f|
## Λ
## .
•SolveSa=Rfor the coefficientsa.
•The quadratized Galerkin approximation isu
## Λ
## Ξ
## =
## N
## ∑
i=1
a
i
b
i
## .
## 79

## Chapter 5
## The Rotation Group
We now specialize to the particular manifold which we will be working on:
the rotation group,SO(3).  It consists of all orthogonal 3×3 matrices with
real entries and determinant 1:
## SO(3) =
## {
## A∈R
## 3×3
## :A
## T
## A=I=AA
## T
,det(A) = 1
## }
## .
In other words, it is the space of all proper rotations ofR
## 3
.  It is a com-
pact Lie group of dimension 3.  As such, it has a unique bi-invariant Haar
measure,μ, and we assume that it is normalized; i.e.μ(SO(3)) = 1.
## 5.1    Parametrizations
There  are  various  parameterizations  ofSO(3);  we  briefly  present  a  few.
The first is referred to asEuler angles.  Euler showed that everyA∈SO(3)
can be factored as
A=E(φ
## 1
## ,θ,φ
## 2
## ) :=R
z
## (φ
## 1
## )R
x
(θ)R
z
## (φ
## 2
## ),
## 80

whereR
x
(α) andR
z
(α) are rotations of angleαabout thex- andz-axes,
respectively:
## R
x
## (α) =
## 
## 
## 
## 
## 
## 100
0    cosα−sinα
0    sinαcosα
## 
## 
## 
## 
## 
## R
z
## (α) =
## 
## 
## 
## 
## 
cosα−sinα0
sinαcosα0
## 001
## 
## 
## 
## 
## 
## .
This is the so-called ZXZ Euler angle decomposition.  Other kinds exist, for
example XYZ, YZY, etc.  For rotations about they-axis, we have
## R
y
## (α) =
## 
## 
## 
## 
## 
cosα0−sinα
## 010
sinα0cosα
## 
## 
## 
## 
## 
## .
For a functionf:SO(3)→R, the Haar integral can be rewritten as
## ∫
## SO(3)
f(x)dμ(x) =
## 1
## 8π
## 2
## ∫
## 2π
## 0
## ∫
π
## 0
## ∫
## 2π
## 0
f(φ
## 1
## ,θ,φ
## 2
) sinθdφ
## 1
dθdφ
## 2
## ,
where we writef(φ
## 1
## ,θ,φ
## 2
) to meanf(E(φ
## 1
## ,θ,φ
## 2
)).  Euler angles, it should
be  noted,  are  not  unique;  for  example,  ifθ=  0  thenE(φ
## 1
## ,θ,φ
## 2
## )  =
## R
z
## (φ
## 1
## +φ
## 2
) can be expressed in any number of ways.
Next,  we  have  theaxis-angleparametrization.   Every  elementAof
SO(3)  has  1  as  an  eigenvalue;  i.e.   an  axis  that  it  keeps  fixed.   A  cor-
responding  unit  eigenvectorvis  called  anaxisforA.   In  the  planev
## ⊥
## ,
vectors rotated byAremain inv
## ⊥
,  and the angle by which they are ro-
tated is called theangleofA, and denotedω(A).  A formula for the angle
of a rotationAis
ω(A) = acos
## (
tr(A)−1
## 2
## )
## .
## 81

Axis-angle parameterizations are not unique; for example, taking the axis
to  be−vand  the  angle  to  be−αalso  yieldsA.   Ifv=  [v
## 1
## ,v
## 2
## ,v
## 3
## ]
## T
and
α∈(−π,π], then the matrix with axis span(v) and angleαis
## A=
## 
## 
## 
## 
## 
v
## 2
## 1
(1−c) +c   v
## 1
v
## 2
## (1−c) +v
## 3
s  v
## 1
v
## 3
## (1−c)−v
## 2
s
v
## 1
v
## 2
## (1−c)−v
## 3
s   v
## 2
## 2
(1−c) +c   v
## 2
v
## 3
## (1−c) +v
## 1
s
v
## 1
v
## 3
## (1−c) +v
## 2
s  v
## 2
v
## 3
## (1−c)−v
## 1
s   v
## 2
## 3
## (1−c) +c
## 
## 
## 
## 
## 
## ,
wherec= cosαands= sinα.
Our next parameterizations is via quaternions.  Aquarternionis a num-
ber of the formw+xi+yj+zk, wherew,x,y, andzare real numbers andi,
j,kare the “purely imaginary” quaternions, satisfyingi
## 2
## =j
## 2
## =k
## 2
## =−1,
ij=k,jk=i,ki=j,ji=−k,kj=−i, andik=−j.  There is a 2-to-1
homomorphism from the group of unit quaternions toSO(3), a unit quater-
nion being a quaternionw+xi+yj+zkfor whichw
## 2
## +x
## 2
## +y
## 2
## +z
## 2
## = 1.
The unit quaternionw+xi+yj+zkmaps to the matrix
## A=
## 
## 
## 
## 
## 
## 1−2y
## 2
## −2z
## 2
## 2xy−2wz2xy−2wy
## 2xy+ 2wz1−2x
## 2
## −2z
## 2
## 2yz−2wx
## 2xz−2wy2yz+ 2wx1−2x
## 2
## −2y
## 2
## 
## 
## 
## 
## 
## .
This may seem to suggest thatSO(3) is 4-dimensional, but, the set of unit
quaternions is in fact 3-dimensional, as it naturally identifies with the unit
sphere inR
## 4
## .
The last parameterizations we will see is via the matrix exponential. The
Lie algebra ofSO(3) isso(3), the space of real skew-symmetric matrices.
## 82

If we set
## X
## 1
## =
## 
## 
## 
## 
## 
## 0    00
## 0    0−1
## 0    10
## 
## 
## 
## 
## 
## ,X
## 2
## =
## 
## 
## 
## 
## 
## 00    1
## 00    0
## −1    0    0
## 
## 
## 
## 
## 
## ,X
## 3
## =
## 
## 
## 
## 
## 
## 0−1    0
## 100
## 000
## 
## 
## 
## 
## 
## ,
then  everyX∈so(3)  can  be  expressed  asα
## 1
## X
## 1
## +α
## 2
## X
## 2
## +α
## 3
## X
## 3
with
α
## 1
## ,α
## 2
## ,α
## 3
∈R. Thematrix exponential, exp :so(3)→SO(3) is an isometry,
where in general for a 3×3 matrixX
exp(X) =
## ∞
## ∑
n=0
## 1
n!
## X
n
## .
Direct  calculations  show  that  exp(αX
## 1
## )  =R
x
(α),  exp(αX
## 2
## )  =R
y
## (α),
and exp(αX
## 3
## ) =R
z
(α);  however,  sinceX
## 1
## ,X
## 2
,  andX
## 3
don’t commute,
calculating  exp(α
## 1
## X
## 1
## +α
## 2
## X
## 2
## +α
## 3
## X
## 3
)  is  complicated  in  general.   If,  on
the  other  hand,  ifu=  [α
## 1
## ,α
## 2
## ,α
## 3
## ]
## T
is  a  unit  vector  inR
## 3
and  we  let
## X=α
## 1
## X
## 1
## +α
## 2
## X
## 2
## +α
## 3
## X
## 3
, then forθ∈R, exp(θX) has axisuand angle
θ, and the Rodrigues addition formula gives
exp(θX) =I+ sinθX+ (1−cosθ)X
## 2
## ,
which is where the parameterizations above for axis-angle comes from.
## 5.2    Harmonic Analysis
Much of the material in this section can be found in [26] for the specific
case  ofSO(3)  and  in  [10]  for  the  general  case.   Being  a  compact  group,
the Peter-Weyl Theorem (see [26, Theorem 2.24] or [10, Theorem 5.12], for
example)  guarantees  that  the  matrix  elements  of  the  irreducible  unitary
## 83

representations  ofSO(3)  form  a  complete  orthogonal  set  forL
## 2
## (SO(3)).
These are theWigner-Dfunctions:  for each`∈Nthere is a unique (up to
unitary equivalence) unitary representationD
## `
which has dimension 2`+ 1.
Forj,k∈{−`,...,`}the Wigner-D functions are the matrix elementsD
## `
j,k
ofD
## `
.  A different (unitarily equivalent) choice forD
## `
will produce different
Wigner-D functions, but their span will be the same.
There are many different formulations for the Wigner-D functionsD
## `
j,k
## .
One  of  the  more  common  can  be  found  in  [26]  and  [17],  given  in  Euler
angles.  They are:
## D
## `
j,k
## (φ
## 1
## ,θ,φ
## 2
## ) =e
## −jφ
## 1
## P
## `
j,k
## (cosθ)e
## −ikφ
## 2
## ,
whereP
## `
j,k
is given by
## P
## `
j,k
(t) =C(1−t)
## −
k−j
## 2
## (1 +t)
## −
j+k
## 2
d
## `−k
dt
## `−k
## [
## (1−t)
## `−j
## (1 +t)
## `+j
## ]
andC=
## (−1)
## `−j
i
k−j
## 2
## `
## (`−j)!
## √
## (1−j)!(`+k)!
## (`+j)!(`−k)!
.  In this formulation, the Wigner-D func-
tions are complex-valued.
There are also real-valued Wigner-D functions found in [20], given by
## D
## `
j,k
## (φ
## 1
## ,θ,φ
## 2
) = sgn(k)Ψ
j
## (φ
## 1
## )Ψ
k
## (φ
## 2
## )
d
## `
## |k|,|j|
## (θ) + (−1)
j
d
## `
## |j|,−|k|
## (θ)
## 2
−sgn(j)Ψ
## −j
## (φ
## 1
## )Ψ
## −k
## (φ
## 2
## )
d
## `
## |k|,|j|
## (θ)−(−1)
j
d
## `
## |j|,−|k|
## (θ)
## 2
## ,
where
## Ψ(x) =
## 
## 
## 
## 
## 
## 
## 
## 
## 
## √
2 cos(jx)ifm >0
## 1ifm= 0
## √
2 sin(|j|x)    ifm <0
## 84

and the little Wigner-D functions are
d
## `
j,k
## (t) =
## √
## (`+k)!(`−k)!
## (`+j)!(`−j)!
## (
sin
t
## 2
## )
k−j
## (
cos
t
## 2
## )
j+k
## P
## (`−k)
## (k−j,j+k)
## (cost),
withP
## (·)
## (·,·)
the Jacobi polynomials.
For each`∈Nandj,k∈ {−`,...,`},D
## `
j,k
is an eigenfunction of−∆
corresponding to the eigenvalueλ
## `
=`(`+ 1).  We refer to`as thedegree
ofD
## `
j,k
andjandkas theorders.  We denote by Π
## L
the set of Wigner-D
functions of degreeLor less. These are our auxiliary spaces for interpolation
with conditionally positive definite kernels onSO(3).
## 5.3    Orthonormal Basis Eigenfunctions
The  Wigner-D  functions  are  orthogonal  inL
## 2
,  but  not  orthonormal.   In
order to be able to express our kernels in the form (2.2.1), we must take
φ
## `
j,k
## =
## D
## `
j,k
## ∥
## ∥
## ∥
## D
## `
j,k
## ∥
## ∥
## ∥
## L
## 2
## =
## √
## 2`+ 1D
## `
j,k
## .
In fact, it is part of the Peter-Weyl Theorem that these form an orthonormal
basis forL
## 2
.  For a functionf∈L
## 2
(SO(3)),`∈Nandj,k∈{−`,...,`}, if
we define
## ̂
f
## `
j,k
## =
## ∫
## SO(3)
fφ
## `
j,k
dμ=
## √
## 2`+ 1
## ∫
## SO(3)
fD
## `
j,k
dμ,
then
f=
## ∞
## ∑
## `=0
## `
## ∑
j,k=−`
## ̂
f
## `
j,k
φ
## `
j,k
## =
## ∞
## ∑
## `=0
## √
## 2`+ 1
## `
## ∑
j,k=−`
## ̂
f
## `
j,k
## D
## `
j,k
## .
## 85

Of particular interest to us are functions that depend only on the ro-
tation angle, calledclass  functions.  These are spanned by thecharacters
c
## `
= tr
## (
## D
## `
## )
.  Thus, for a class functionf:SO(3)→Rthere is a uniquely
determined
## ̃
f: [0,π]→Rfor whichf(x) =
## ̃
f(ω(x)) for allx∈SO(3).  For
such functions the Haar integral simplifies to
## ∫
## SO(3)
f(x)dμ(x) =
## 2
π
## ∫
π
## 0
## ̃
f(t) sin
## 2
## (
t
## 2
## )
dt.
We also have the following addition formula for Wigner-D functions:
## `
## ∑
j,k=−`
## D
## `
j,k
(x)D
## `
j,k
(y) = tr
## (
## D
## `
(x)D
## `
## (y
## −1
## )
## )
= tr
## (
## D
## `
## (
y
## −1
x
## ))
## =c
## `
## (
y
## −1
x
## )
## =U
## 2`
## (
cos
## (
ω
## (
y
## −1
x
## )
## 2
## ))
## ,
whereU
n
is  the  Chebyshev  polynomial  of  the  second  kind  of  degreen.
The representation corresponding to our complete orthonormal set isφ
## `
## =
## √
## 2`+ 1D
## `
, and the addition formula for them is
## `
## ∑
j=−`
φ
## `
j,k
## (x)φ
## `
j,k
## (y) =
## √
## 2`+ 1U
## 2`
## (
cos
## (
ω
## (
y
## −1
x
## )
## 2
## ))
## .
We finish this section with lemmas that give theH
m
norm of our nor-
malized eigenfunctions and their covariant derivatives.
Lemma 5.3.1.Forτ∈N,
## ∥
## ∥
## ∥
φ
## `
j,k
## ∥
## ∥
## ∥
## H
τ
## = (1 +λ
## `
## )
τ/2
## .
## Proof.
## ∥
## ∥
φ
## `
j,k
## ∥
## ∥
## 2
## H
τ
## =
## 〈
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## H
τ
## =
## 〈
## (I−∆)
τ
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## L
## 2
## = (1 +λ
## `
## )
τ
## 〈
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## L
## 2
## = (1 +λ
## `
## )
τ
## .
## 86

Lemma 5.3.2.Forτ∈N,
## ∥
## ∥
## ∥
## ∇φ
## `
j,k
## ∥
## ∥
## ∥
## H
τ
## ≤C`
τ+1
## .
Proof.By  Lemma  2.5.3,  there  are  numbersa
n
,n=  0,...2τsuch  that
## (I−∆)
τ
## =
## ∑
## 2τ
n=0
a
n
## (∇
n
## )
## ∗
## ∇
n
## .  Thus,
## ∥
## ∥
## ∇φ
## `
j,k
## ∥
## ∥
## 2
## H
τ
## =
## 〈
## ∇φ
## `
j,k
## ,φ
## `
j,k
## 〉
## H
τ
## =
## 〈
## (I−∆)
τ
## ∇φ
## `
j,k
## ,∇φ
## `
j,k
## 〉
## L
## 2
## =
## 2τ
## ∑
n=0
a
n
## 〈
## ∇
n+1
φ
## `
j,k
## ,∇
n+1
φ
## `
j,k
## 〉
## L
## 2
## =
## 2τ
## ∑
n=0
a
n
## 〈
## (
## ∇
n+1
## )
## ∗
## ∇
n+1
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## L
## 2
## .
By Lemma 2.5.2 there are numbersb
i
,i= 0,...2τ+ 1 such that
## 2τ
## ∑
n=0
a
n
## (
## ∇
n+1
## )
## ∗
## ∇
n+1
## =
## 2τ+1
## ∑
i=0
b
i
## ∆
i
## .
## Thus,
## ∥
## ∥
## ∇φ
## `
j,k
## ∥
## ∥
## 2
## H
τ
## =
## 2τ+1
## ∑
i=0
b
i
## 〈
## ∆
i
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## L
## 2
## =
## 2τ+1
## ∑
i=0
b
i
## (−1)
i
λ
i
## `
## 〈
φ
## `
j,k
## ,φ
## `
j,k
## 〉
## L
## 2
## ≤C`
## 2τ+2
where we have takenC= max
i∈{1,...,2τ+1}
## |b
i
|.  Taking square roots yields
the result.
5.4    Kernels and Spaces
We recall here certain definitions from Chapters 2 and 4, restated using no-
tation better suited forSO(3).  We nowdoassume that the eigenvaluesλ
## `
## 87

are distinct; our Hilbert-Schmidt series will have a form similar to (2.2.1),
except now we group the basis eigenfunctions for the same eigenvalue to-
gether.  Thus, our kernels Φ :SO(3)×SO(3)−→Rhave the form
## Φ(x,y) =
## ∞
## ∑
## `=0
## ̂
φ(`)
## `
## ∑
j,k=−`
φ
## `
j,k
## (x)φ
## `
j,k
## (y).(5.4.1)
We  assume  the  coefficients  satisfy  (2.8.1)  for  someτ∈Nwithτ >3/2,
either for all`∈Nif Φ is positive definite, or for all` > Lif Φ is condi-
tionally positive definite with respect to Π
## L
.  This ensures that the native
space norm is equivalent to the Sobolev norm in the positive definite case.
The native space for Φ is
## N
## Φ
## =
## 
## 
## 
f∈L
## 2
## :
## ∞
## ∑
## `=0
## ̂
φ(`)
## −1
## `
## ∑
j,k=−`
## ∣
## ∣
## ∣
## ̂
f
## `
j,k
## ∣
## ∣
## ∣
## 2
## <∞
## 
## 
## 
## ,
with inner product
## 〈f,g〉
## N
## Φ
## =
## ∞
## ∑
## `=0
## ̂
φ(`)
## −1
## `
## ∑
j,k=−`
## ̂
f
## `
j,k
## ̂g
## `
j,k
## .
The Sobolev spaceH
τ
is
## H
τ
## =
## 
## 
## 
f∈L
## 2
## :
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
τ
## `
## ∑
j,k=−`
## ∣
## ∣
## ∣
## ̂
f
## `
j,k
## ∣
## ∣
## ∣
## 2
## <∞
## 
## 
## 
## ,
with inner product
## 〈f,g〉
## H
τ
## =
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
τ
## `
## ∑
j,k=−`
## ̂
f
## `
j,k
## ̂g
## `
j,k
## .
## 88

For a subset Ω⊂SO(3), we also have the Sobolev space
## W
m
## 2
## (Ω) =
## {
f∈L
## 2
## (Ω) :
m
## ∑
k=0
## ∫
## Ω
## 〈
## ∇
k
f,∇
k
f
## 〉
p
dμ(p)<∞
## }
## ,
with inner product
## 〈f,g〉
## W
m
## 2
## (Ω)
## =
m
## ∑
k=0
## ∫
## Ω
## 〈
## ∇
k
f,∇
k
g
## 〉
p
dμ(p).
SinceSO(3)  is  compact  and  without  boundary,  when  Ω  is  all  ofSO(3)
andτ=mis an integer,H
m
## =W
m
## 2
with the norms being equivalent by
Proposition 2.7.3.  If in addition (2.8.1) holds for all`∈N; i.e.  Φ is positive
definite, thenN
## Φ
## =H
m
## =W
m
## 2
, with all three norms being equivalent.
## 5.5    Energy Estimates:  Two Examples
We  now  exhibit  two  examples  of  kernels  onSO(3)  that  provide  energy
estimates.  They both have advantages and drawbacks.  The first has the
advantage that a closed-form formula for it is known, but has the drawback
that  it  only  provides  an  algebraic  energy  estimate.   The  second  has  the
advantage  that  it  provides  an  exponential  energy  estimate,  but  has  the
drawback that no closed-form formula for it is known.
Example 5.5.1.An example of a family of kernels of the form (5.4.1) are
therotational surface splines:  form≥3,
## Φ
m
## (x,y) =
## (
sin
## (
ω
## (
y
## −1
x
## )
## 2
## ))
## 2m−3
## .
## 89

The coefficients of Φ
m
, found in [17], are
## ̂
φ
m
## (`) =
## (2m−2)!
π(−4)
m−1
## (2`+ 1)
m−1
## ∏
j=0
## 1
## `(`+ 1)−
## (
j
## 2
## −
## 1
## 4
## )
## .
and Φ
m
is conditionally positive definite with respect to Π
m−2
.  The coeffi-
cients
## ̂
φ
m
(`) satisfy (2.8.1) withτ=m−
## 1
## 2
.  The corresponding differential
operator, for which Φ
m
is the fundamental solution, is
## L=
## (2m−2)!
π(−4)
m−1
m−1
## ∏
j=0
## (
## ∆−
## (
j
## 2
## −
## 1
## 4
## ))
## .
This operator doesnotannihilate Π
m−2
, and so Φ
m
provides only an alge-
braic energy estimate.
Let  Ξ  and  Λ  be  Π
m−2
-unisolvent  sets  of  centers  with  #Ξ<#Λ  and
h
## Λ
< h
## Ξ
, and consider using Φ
m
to produce the quadratized Galerkin ap-
proximationu
## Λ
## Ξ
to the weak solution ofLu=f, withf∈H
m
, using Ξ as
the set of centers for Galerkin approximation and Λ the set of centers for
quadrature.  Then Corollary 4.2.7 guarantees that
## ∥
## ∥
u−u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +
## (
h
## Λ
h
## Ξ
## )
m
h
## −11/2
## Ξ
h
## −5/2
## Λ
## )
## ‖f‖
## H
s
## ,
wheresis  determined  as  in  Theorem  4.2.1,  and  Remark  4.2.8  shows  we
should take our oversampling exponent to bep= 2 +
## 19
## 2m−5
## .
Example 5.5.2.Another example of a family of kernels of the form (5.4.1)
are the “ideal” kernels
κ
m
## (x,y) =
## ∞
## ∑
## `=0
## (1 +λ
## `
## )
## −m
## `
## ∑
j,k=−`
φ
## `
j,k
## (x)φ
## `
j,k
## (y).
Note that we have not writtenφ
## `
j,k
(y) - we assume from here on out that
## 90

the  Wigner-D  functions,  and  hence  our  orthonormal  eigenfunctions,  are
real-valued.   Here  (2.8.1)  holds  withC
## 1
## =C
## 2
=  1  andτ=m,  since
the coefficients ofκ
m
are exactlŷκ
m
## (`) = (1 +λ
## `
## )
## −m
.  The corresponding
operatorLfor whichκ
m
is the fundamental solution isL= (I−∆)
m
## .  Since
## ̂κ
m
(`) is positive for all`,κ
m
is positive definite, and therefore provides an
exponential energy estimate.
Let Ξ, Λ, andf∈H
m
be as in Example 5.5.1.  Ifu
## Λ
## Ξ
is the quadratized
Galerkin  approximation  to  the  solution  ofLu=f,  then  Corollary  4.2.7
guarantees that
## ∥
## ∥
u−u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +
## (
h
## Λ
h
## Ξ
## )
m
h
## −11/2
## Ξ
h
## −1
## Λ
## |log (h
## Λ
## )|
## 3
## )
## ‖f‖
## H
m
## ,
and Remark 4.2.8 shows that we should take our oversampling exponent to
bep=
## 3
log(m)
## W
## (
## 1
## 3
log(m)h
m+1
## Ξ
## |log (h
## Ξ
## )|
## −1
## )
.  Again, taking the oversam-
pling exponent that would arise in the algebraic case will suffice, so, we can
actually takep= 2 +
## 19
## 2m−5
## .
Although  the  kernel  from  Example  5.5.2  doesn’t  have  a  closed  form,
we  can  still  use  it.   In  the  next  chapter,  we  show  how  to  truncate  it  to
get a Galerkin approximation.  What we must determine, then, iswhento
truncate.
## 5.6    Algorithm:  Quadrature Weights
To implement our Galerkin approximation scheme, we need one final algo-
rithm, one for computing the quadrature weights.  We restrict our attention
to the kernel Φ
m
from Section 5.5.  As per Section 3.1, we need to know
the integrals
## ∫
## SO(3)
## Φ
m
## (·,y)dμand
## ∫
## SO(3)
φ
## `
j,k
dμ.  The fact that the former
integral is independent ofyfollows from the fact that Φ
m
has a Hilbert-
## 91

Schmidt decomposition and the orthonormality of the eigenfunctions of the
Laplace-Beltrami operator.  Hence, we can takey= Id to obtain
## J=
## ∫
## SO(3)
Φ(·,Id)dμ=
## 2
π
## ∫
π
## 0
## (
sin
## (
ω
## 2
## ))
## 2m−1
dω=
2Γ(m)
## √
πΓ(m+
## 1
## 2
## )
## .
## When`=j=k=  0,φ
## 0
## 0,0
=  1,  and  so  the  normalization  of  the  Haar
measure  gives
## ∫
## SO(3)
φ
## 0
## 0,0
dμ=  1.   For  any  other  eigenfunctionφ
## `
j,k
,  or-
thonormality gives
## J
## `
j,k
## =
## ∫
## SO(3)
φ
## `
j,k
dμ=
## 〈
φ
## `
j,k
## ,φ
## 0
## 0,0
## 〉
## L
## 2
## = 0.
This leads us to the following.
Algorithm 4Quadrature Weights
## Input:  Λ ={ζ
## 1
## ,...,ζ
## M
}, kernel parameterm.
Output:  Quadrature weightsw={w
## 1
## ,...,w
## M
## }.
•Form the auxiliary matrix
## P=
## [
φ
## `
j,k
## ∣
## ∣
## Λ
## ]
## `∈{0,...,m−2},j,k∈{−`,`}
## .
•Form the collocation matrixK
## Ξ
## .
•solve the system
## [
## K
## Ξ
## P
## P
## T
## 0
## ][
w
v
## ]
## =
## [
## J1
u
## ]
## ,
whereJ=
2Γ(m)
## √
πΓ(m+
## 1
## 2
## )
,1is theM×1 vector of 1’s, anduis theQ×1
vector whose first entry is 1 and all other entries are 0 (Q=
## (
## 2m−1
## 3
## )
is
the dimension of the auxiliary space Π
m−2
## ).
•The quadrature weights arew.
## 92

## Chapter 6
A TruncatedSO(3)Kernel
The rotational surface splines have the advantage of having a closed form.
The  same  can  not  be  said  for  some  other  kernels  onSO(3),  such  as  the
one in Example 5.5.2.  In practice, to use those kinds of kernels, some sort
of truncation of the uniformly convergent series in (5.4.1) is desirable.  We
explore in this chapter the error that arises when such truncation occurs,
in a specific case.
Our strategy is as follows.  We employ a particular kernelκ
m
## ,m >5/2
with an expansion as in (5.4.1), and which provides an exponential energy
estimate.  We then truncate by taking only finitely many terms in (5.4.1) to
get the truncated kernel ̃κ
m
, the number of terms beingN, thetruncation
parameter.  We then form the truncated Lagrange basis{ ̃χ
ξ
## }
ξ∈Ξ
, consisting
of the Lagrange functions for the truncated kernel.  The truncated stiffness
matrix in the truncated Lagrange basis,
## ̃
## B
## Ξ
is then used to obtain the trun-
cated Galerkin approximation, ̃u
## Ξ
## .
Upon  establishing  anL
## 2
truncation  error  estimate  for  the  truncated
Galerkin approximation, we then explore the error that arises by quadra-
## 93

tizing  the  truncated  Galerkin  approximation.   This  means  replacing  the
entries of the truncated stiffness matrix with quadrature estimates to form
the quadratized truncated stiffness matrix
## ̃
## B
## Λ
## Ξ
, which is then used to obtain
the quadratized truncated Galerkin approximation, ̃u
## Λ
## Ξ
.  The full error es-
timate is then attained via repeated applications of the triangle inequality:
## ∥
## ∥
u− ̃u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤‖u−u
## Ξ
## ‖
## L
## 2
## +‖u
## Ξ
## − ̃u
## Ξ
## ‖
## L
## 2
## +
## ∥
## ∥
## ̃u
## Ξ
## − ̃u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## .
We restrict our attention to the particular family of kernels from Ex-
ample 5.5.2; namely,
κ
m
## (x,y) =
## ∞
## ∑
## `=0
## ̂κ
m
## (`)
## −`
## ∑
j,k=−`
φ
## `
j,k
## (x)φ
## `
j,k
## (y),
with Fourier-Chebyshev coefficients
## ̂κ
m
## (`) = (1 +λ
## `
## )
## −m
## = (1 +`(`+ 1))
## −m
## .
Note that we have not written the conjugate ofφ
## `
j,k
(y) in the expansion of
our kernel.  This is deliberate - to simplify matters, we assume throughout
that the Wigner-D functions, and hence our orthonormal basis functions,
are real-valued.  Also note that the bound on the coefficients ofκ
m
## ,
## |̂κ
m
## (`)|≤C`
## −2m
## ,(6.0.1)
holds for all`≥1.
Sincêκ(`)>0 for all`∈N,κ
m
is positive definite.  As such it provides
an exponential energy estimate, and all our previous results for such kernels
apply.  Letνbe the constant from the energy estimate.  In what follows
whenever we say “there exists a positive constantC”, it is understood that
## 94

Cdepends on the kernel Φ
m
, the mesh ratioρof our centers Ξ, the tensor
aand functionbin our differential equationL, andν.
## 6.1    The Truncated Kernel
Thetruncated kernel, ̃κ
m
, withtruncation parameterN, is
## ̃κ
m
## (x,y) =
## N
## ∑
## `=0
## ̂κ
m
## (`)
## `
## ∑
j,k=−`
φ
## `
j,k
## (x)φ
## `
j,k
## (y).
Our first order of business is to establish pointwise andL
## 2
error estimates
for the truncated kernel, andL
## 2
error estimates for its covariant derivative.
We point out that bothκ
m
and ̃κ
m
, and the covariant derivatives of both,
enjoy anL
## 2
-invariance in the sense that‖κ
m
## (·,x)‖
## L
## 2
## =‖κ
m
## (·,y)‖
## L
## 2
for all
x,y∈SO(3).
Lemma 6.1.1.Form >1andN >0the uniform bound
## |κ
m
## (x,y)− ̃κ
m
(x,y)|≤CN
## 2−2m
## .
holds for allx,y∈SO(3).
Proof.Since
max
θ∈[0,π]
## ∣
## ∣
## ∣
## ∣
## U
## 2`
## (
cos
## (
θ
## 2
## ))
## ∣
## ∣
## ∣
## ∣
## =U
## 2`
## (1) = 2`+ 1,
## 95

we have by (6.0.1) that
## |κ
m
## (x,y)− ̃κ
m
## (x,y)|=
## ∣
## ∣
## ∣
## ∣
## ∣
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)U
## 2`
## (
cos
## (
ω
## (
y
## −1
x
## )
## 2
## ))
## ∣
## ∣
## ∣
## ∣
## ∣
## ≤C
## ∞
## ∑
## `=N+1
## `
## −2m
## (2`+ 1)
## ≤C
## ′
## ∞
## ∑
## `=N+1
## `
## 1−2m
## .
The result now follows from an integral comparison.
Lemma 6.1.2.Form >3/2,τ∈Nwith0≤τ <2m−2,N >0, and any
ζ∈SO(3),
## ‖κ
m
## (·,ζ)− ̃κ
m
## (·,ζ)‖
## H
τ
## ≤CN
## 2+τ−2m
## .
Proof.We need only prove the caseζ= Id.  We have
κ
m
(x,Id)− ̃κ
m
(x,Id) =
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j,k=−`
φ
## `
j,k
## (x)φ
## `
j,k
(Id)
## =
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j=−`
φ
## `
j,j
## (x).
Using (6.0.1) and Lemma 5.3.1, then, we have
## ‖κ
m
(·,Id)− ̃κ
m
(·,Id)‖
## H
τ
## ≤
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j=−`
## ∥
## ∥
φ
## `
j,j
## ∥
## ∥
## H
τ
## ≤C
## ∞
## ∑
## `=N+1
## `
τ−2m
## (2`+ 1)
## ≤C
## ∞
## ∑
## `=N+1
## `
## 1+τ−2m
## .
The result now follows from an integral comparison.
## 96

Lemma 6.1.3.Form >3/2,τ∈Nwith0≤τ <2m−3,N >0, and any
ζ∈SO(3),
## ‖∇(κ
m
## (·,ζ)− ̃κ
m
## (·,ζ))‖
## H
τ
## ≤CN
## 3+τ−2m
## .
Proof.Again, we need only prove the caseζ= Id.  We have
## ∇(κ
m
(x,Id)− ̃κ
m
(x,Id)) =
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j,k=−`
## (
## ∇φ
## `
j,k
## (x)
## )
φ
## `
j,k
(Id)
## =
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j=−`
## ∇φ
## `
j,j
## (x).
and so by (6.0.1) and Lemma 5.3.2,
## ‖∇(κ
m
(·,Id)− ̃κ
m
(·,Id))‖
## H
τ
## ≤
## ∞
## ∑
## `=N+1
## ̂κ
m
## (`)
## `
## ∑
j=−`
## ∥
## ∥
## ∇φ
## `
j,j
## ∥
## ∥
## H
τ
## ≤C
## ∞
## ∑
## `=N+1
## `
## 1+τ−2m
## (2`+ 1)
## ≤C
## ∞
## ∑
## `=N+1
## `
## 2+τ−2m
## .
The result now follows from an integral comparison.
## 6.2    The Truncated Lagrange Basis
To get thetruncated Lagrange basis,{ ̃χ
ξ
## }
ξ∈Ξ
, we take thetruncated collo-
cation matrix,
## ̃
## K
## Ξ
## ={ ̃κ
m
## (ξ,η)}
ξη∈Ξ
, and the coefficients ̃α
ξ
## ={ ̃α
ξ,η
## }
η∈Ξ
of
## ̃χ
ξ
## =
## ∑
η∈Ξ
## ̃α
ξ,η
## ̃κ
m
(·,η) are obtained by solving the linear system
## ̃
## K ̃α
ξ
## =
δ
ξ
.    We  reiterate  that  the  choice  of  basis  does  not  affect  interpolants,
Galerkin  approximations,  or  quadratized  Galerkin  approximation.   It  is
used purely for the stability of the approximation scheme.
The first result in this section concerns the 1-norm of the inverse of the
## 97

collocation matrix. In this result and in many that follow, we are computing
the 1-norm of a self-adjoint matrix.  We note that for a self-adjoint matrix
## A,‖A‖
## 2
## ≤‖A‖
## 1
, and so each such result has a corresponding corollary for
the 2-norm.
Lemma 6.2.1.LetK
## Ξ
be the collocation matrix for interpolation with the
kernelκ
m
on the set of centersΞ, withm >3/2and mesh ratioρ.  There
exists a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
≤Ch
## 3−2m
## Ξ
## .
Proof.The entries in columnξofK
## −1
## Ξ
are precisely the coefficients of the
Lagrange functionχ
ξ
## =
## ∑
η∈Ξ
α
ξη
κ(·,η).  Hence,
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## |α
ξη
## |.
We follow [14] by noting that the proof of [11, Equation 5.6] gives, “mutatis
mutandis”,
## |α
ξζ
|≤Cq
## 3−2m
## Ξ
exp
## (
## −ν
dist(ξ,ζ)
h
## Ξ
## )
with positive constantsCandνthat depend only onk
m
.  These are the
same as the constants from our energy estimate.  We can proceed by divid-
ingSO(3) into annuliA
n
with centerξ, outer radiusnh
## Ξ
, and inner radius
## (n−1)h
## Ξ
## ,n= 1,2,...,n
max
.  We then have
## ∑
ζ∈Ξ
## |α
ξζ
## |=
n
max
## ∑
n=1
## ∑
ζ∈A
n
## |α
ξζ
|≤Cq
## 3−2m
## Ξ
n
max
## ∑
n=1
## ∑
ζ∈A
n
exp
## (
## −ν
dist(ξ,ζ)
h
ξ
## )
## .
EachA
n
has volume∼n
## 2
h
## 3
## Ξ
, and so
## # (Ξ∩A
n
## )∼
n
## 2
h
## 3
## Ξ
q
## 3
## Ξ
## =ρ
## 3
n
## 2
## .
## 98

Also, the minimum distance betweenζ∈A
n
andξis bounded below by
## (n−1)h
## Ξ
, which gives
## ∑
ζ∈Ξ
## |α
ξζ
|≤Cq
## 3−2m
## Ξ
n
max
## ∑
n=1
## # (Ξ∩A
n
) exp
## (
## −ν
## (n−1)h
## Ξ
h
## Ξ
## )
≤Cq
## 3−2m
## Ξ
n
max
## ∑
n=1
n
## 2
e
## −ν(n−1)
≤Cq
## 3−2m
## Ξ
## ∞
## ∑
n=0
## (n+ 1)
## 2
e
## −νn
## .
Summing the series proves the result:
## ∑
ζ∈Ξ
## |α
ξζ
|≤Cq
## 3−2m
## Ξ
e
## 2ν
## (e
ν
## + 1)
## (e
ν
## −1)
## 3
## =C
## ′
q
## 3−2m
## Ξ
## .
We now establish a bound on the 1-norm of the difference between the
collocation matrix and truncated collocation matrix.  Again, since both are
self-adjoint, the corresponding result for the 2-norm follows automatically.
Lemma  6.2.2.LetK
## Ξ
and
## ̃
## K
## Ξ
be  the  collocation  matrix  and  truncated
collocation matrix, respectively, forκ
m
and ̃κ
m
, respectively, withm >5/2
and truncation parameterN >0.  There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## K
## Ξ
## −
## ̃
## K
## Ξ
## ∥
## ∥
## ∥
## 1
≤Ch
## −3
## Ξ
## N
## 2−2m
## .
Proof.By Lemma 6.1.1,
## ∥
## ∥
## ∥
## K
## Ξ
## −
## ̃
## K
## Ξ
## ∥
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## |κ
m
## (ξ,η)− ̃κ
m
(ξ,η)|≤C(#Ξ)N
## 2−2m
## ,
and #Ξ∼h
## 3
## Ξ
## .
## 99

We also need to bound the inverse of the truncated collocation matrix.
To do so requires us to assume that the truncation parameter is at least a
certain size, given in the following lemma.
Lemma  6.2.3.Let
## ̃
## K
## Ξ
be  the  truncated  collocation  matrix  for ̃κ
m
,  with
m >5/2and truncation parameterN.  LetCbe the largest constant from
Lemmas 6.2.1 and 6.2.2.  SupposeNsatisfies
## N≥
## (
2Ch
## −2m
## Ξ
## )
## 1
## 2m−2
## .
There exists a constantC
## ′
such that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## ≤C
## ′
h
## 3−2m
## Ξ
## .
Proof.Note that
## ̃
## K
## Ξ
## =K
## Ξ
## (
## I−K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
## ,
and hence
## ̃
## K
## −1
## Ξ
## =
## (
## I−K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
## −1
## K
## −1
## Ξ
## .
By Lemmas 6.2.1 and 6.2.2, then, along with the condition onN, we have
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## )
## ∥
## ∥
## ∥
## 1
## ≤
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
## ∥
## ∥
## ∥
## K
## Ξ
## −
## ̃
## K
## Ξ
## ∥
## ∥
## ∥
## 1
≤Ch
## −2m
## Ξ
## N
## 2−2m
## ≤
## 1
## 2
## .
This ensures the convergence of the Neumann series
## (
## I−K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
## −1
## =
## ∞
## ∑
n=0
## (
## K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
n
## ,
## 100

and we have
## ∥
## ∥
## ∥
## ∥
## (
## I−K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## ∥
## 1
## ≤
## ∞
## ∑
n=0
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## )
## ∥
## ∥
## ∥
n
## 1
## ≤
## ∞
## ∑
n=0
## (
## 1
## 2
## )
n
## = 2.
Hence by Lemma 6.2.1,
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## ≤
## ∥
## ∥
## ∥
## ∥
## (
## I−K
## −1
## Ξ
## (
## K
## Ξ
## −
## ̃
## K
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## ∥
## 1
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
≤Ch
## 3−2m
## Ξ
## .
Corollary 6.2.4.Adopt the notation and assumptions from Lemma 6.2.3.
There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ‖ ̃χ
ξ
## ‖
## H
m
≤Ch
## 3−2m
## Ξ
## .
Proof.It is straightforward to show that‖ ̃κ
m
## (·,η)‖
## H
m
is bounded for each
η∈SO(3) byC=
## √
1 + 3ζ(2m−1), whereζis the Riemann zeta function.
## Now,
## ‖ ̃χ
ξ
## ‖
## H
m
## =
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## ̃α
ξ,η
## ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## ≤
## ∑
η∈Ξ
## | ̃α
ξ,η
## |‖ ̃κ
m
## (·,η)‖
## H
m
## ≤C
## ∑
η∈Ξ
## | ̃α
ξ,η
|=C‖ ̃α
ξ
## ‖
## `
## 1
## (Ξ)
## =C
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
δ
ξ
## ∥
## ∥
## ∥
## `
## 1
## (Ξ)
## ≤C
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## ,
and the result now follows from Lemma 6.2.3.
We now turn to a result that is analogous to Theorem 4.1.5.  That the-
orem was used to bound the entries, and then the 1-norm, of the difference
## 101

between the stiffness matrix and the quadratized stiffness matrix.  This re-
sult will be used in the next section to bound the entries of the difference
between the stiffness matrix and the truncated stiffness matrix as a func-
tion of their distance from the main diagonal.  That will then be used to
get a bound on the 1-norm of that difference.
Theorem 6.2.5.Let{ ̃χ
ξ
## }
ξ∈Ξ
be the truncated Lagrange basis for ̃κ
m
, with
m >5/2.  Thenb ̃χ
ξ
## ̃χ
η
## ∈H
m
## ∩L
## ∞
anda(∇ ̃χ
ξ
## ,∇ ̃χ
η
## )∈H
m−1
## .  Moreover,
there is a constantCsuch that, forh
## Ξ
sufficiently small,
## ‖b ̃χ
ξ
## ̃χ
η
## ‖
## H
m
≤Ch
## 3−2m
## Ξ
and
## ∥
## ∥
a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
## )
## ∥
## ∥
## H
m−1
≤Ch
## 2−m
## Ξ
## .
Proof.The proof of this result is identical to the proof of Theorem 4.1.5,
except that we use the bound from Corollary 6.2.4 instead of a bump esti-
mate.
We  end  this  section  with  results  that  bound  the  truncation  error  for
Lagrange  functions  and  their  covariant  derivatives.   They  will  be  instru-
mental  in  proving  truncation  error  estimates  for  the  truncated  Galerkin
approximation.
Lemma 6.2.6.Let{χ
ξ
## }
ξ∈Ξ
and{ ̃χ
ξ
## }
ξ∈Ξ
be the Lagrange basis and trun-
cated  Lagrange  basis,  respectively,  withm >5/2.   Suppose  the  truncation
parameterNsatisfies the lower bound in Lemma 6.2.3.  There is a constant
## C
## ′
such that, forh
## Ξ
sufficiently small,
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## L
## 2
## ≤C
## ′
h
## 3
## 2
## −4m
## Ξ
## N
## 2−2m
## .
## 102

Proof.By  “smuggling  in”  the  “hybrid”  function
## ∑
η∈Ξ
α
ξη
̃κ(·,η),  which
pairs  the  coefficients  of  the  Lagrange  functions  with  the  corresponding
translates of the truncated kernel, we have
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## L
## 2
## =
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
κ
m
## (·,η)−
## ∑
η∈Ξ
## ̃α
ξη
## ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## ≤
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## +
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## .
## (6.2.7)
We handle the two summands in (6.2.7) separately.  For the first, H ̈older’s
inequality for sums gives
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
## =
## ∫
## SO(3)
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## ∑
η∈Ξ
α
ξη
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∣
## ∣
## ∣
## ∣
## ∣
## ∣
## 2
dμ
## ≤
## ∫
## SO(3)
## 
## 
## ∑
η∈Ξ
## |α
ξη
## ||κ
m
## (·,η)− ̃κ
m
## (·,η)|
## 
## 
## 2
dμ
## ≤
## ∫
## SO(3)
## 
## 
## 
## 
## 
## ∑
η∈Ξ
## |α
ξη
## |
## 2
## 
## 
## 1/2
## 
## 
## ∑
ζ∈Ξ
## |κ
m
## (·,ζ)− ̃κ
m
## (·,ζ)|
## 2
## 
## 
## 1/2
## 
## 
## 
## 2
dμ
## =
## ∑
η∈Ξ
## |α
ξη
## |
## 2
## ∑
ζ∈Ξ
## ‖κ
m
## (·,ζ)− ̃κ
m
## (·,ζ)‖
## 2
## L
## 2
## (6.2.8)
We  bound  the  two  factors  in  (6.2.8)  separately.   For  the  first,  note  that
## 103

sinceK
## −1
## Ξ
is self-adjoint,
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
.  Thus, Lemma 6.2.1 gives
## ∑
η∈Ξ
## |α
ξη
## |
## 2
## =‖α
ξ
## ‖
## 2
## `
## 2
## (Ξ)
## =
## ∥
## ∥
## K
## −1
## Ξ
δ
ξ
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## ≤
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 2
## 2
## ≤
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 2
## 1
≤Ch
## 6−4m
## Ξ
## .
## (6.2.9)
For the second factor in (6.2.8), we have by Lemma 6.1.2 that
## ∑
ζ∈Ξ
## ‖κ
m
## (·,ζ)− ̃κ
m
## (·,ζ)‖
## 2
## L
## 2
## ≤(#Ξ)CN
## 3−4m
≤Cq
## −3
## Ξ
## N
## 4−4m
## .(6.2.10)
## Usingq
## −3
## Ξ
## =ρ
## 3
h
## −3
## Ξ
,  putting  (6.2.9)  and  (6.2.10)  into  (6.2.8),  and  taking
square roots, we arrive at
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
## 3
## 2
## −2m
## Ξ
## N
## 2−2m
## .(6.2.11)
We now turn to the second summand in (6.2.7).  Similar considerations give
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
## ≤
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |
## 2
## ∑
ζ∈Ξ
## ‖ ̃κ
m
## (·,ζ)‖
## 2
## L
## 2
## .
## (6.2.12)
As before, we bound the two factors in (6.2.12) separately.  For the first,
we start with
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |
## 2
## =‖α
ξ
## − ̃α
ξ
## ‖
## 2
## `
## 2
## (Ξ)
## =
## ∥
## ∥
## ∥
## (
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## )
δ
ξ
## ∥
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## ≤
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## 2
## 104

BothK
## −1
## Ξ
and
## ̃
## K
## −1
## Ξ
are self-adjoint, and so
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## .
Also,  we  haveK
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## =
## ̃
## K
## −1
## Ξ
## (
## ̃
## K
## Ξ
## −K
## Ξ
## )
## K
## −1
## Ξ
.   Hence  by  Lemmas
6.2.1, 6.2.2, and 6.2.3,
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |
## 2
## ≤
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## 1
## ≤
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## 1
## ∥
## ∥
## ∥
## ̃
## K
## Ξ
## −K
## Ξ
## ∥
## ∥
## ∥
## 2
## 1
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 2
## 1
≤Ch
## 6−8m
## Ξ
## N
## 4−4m
## .
## (6.2.13)
For the second factor in (6.2.12), absorb‖ ̃κ
m
## (·,ζ)‖
## 2
## L
## 2
, which depends only
onm, into the constant.  This gives us
## ∑
ζ∈Ξ
## ‖ ̃κ
m
## (·,ζ)‖
## 2
## L
## 2
≤C(#Ξ)≤Cq
## −3
## Ξ
## .(6.2.14)
Again usingq
## −3
## Ξ
## =ρ
## 3
h
## −3
## Ξ
, putting (6.2.13) and (6.2.14) into (6.2.12), and
taking square roots, we arrive at
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,ζ)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
## 3
## 2
## −4m
## Ξ
## N
## 2−2m
## .(6.2.15)
Finally, then, putting (6.2.11) and (6.2.15) into (6.2.7) yields the result.
An almost identical proof, where we use Lemma 6.1.3 instead of 6.1.2,
yields the following.
Lemma 6.2.16.Let{χ
ξ
## }
ξ∈Ξ
and{ ̃χ
ξ
## }
ξ∈Ξ
be the Lagrange basis and trun-
cated  Lagrange  basis,  respectively,  forκ
m
and ̃κ
m
,  respectively,  withm >
5/2.    Suppose  the  truncation  parameterNsatisfies  the  lower  bound  in
## 105

Lemma 6.2.3.  There is a constantC
## ′
such that, forh
## Ξ
sufficiently small,
## ‖∇(χ
ξ
## − ̃χ
ξ
## )‖
## L
## 2
## ≤C
## ′
h
## −4m
## Ξ
## N
## 3−2m
We also need an analogous result for the Sobolev norm of the difference
between the Lagrange function and the truncated Lagrange function.
Lemma 6.2.17.Let{χ
ξ
## }
ξ∈Ξ
and{ ̃χ
ξ
## }
ξ∈Ξ
be the Lagrange basis and trun-
cated  Lagrange  basis,  respectively,  withm >5/2.   Suppose  the  truncation
parameterNsatisfies the lower bound in Lemma 6.2.3.  There is a constant
## C
## ′
such that, forh
## Ξ
sufficiently small,
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## H
m
## ≤C
## ′
h
## −4m
## Ξ
## N
## 2−2m
## .
Proof.By  “smuggling  in”  the  “hybrid”  function
## ∑
η∈Ξ
α
ξη
̃κ(·,η),  which
pairs  the  coefficients  of  the  Lagrange  functions  with  the  corresponding
translates of the truncated kernel, we have
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## H
m
## =
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
κ
m
## (·,η)−
## ∑
η∈Ξ
## ̃α
ξη
## ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## ≤
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
α
ξη
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## +
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## .
## (6.2.18)
We handle the sums in (6.2.18) separately.  For the first, we’ve already seen
## 106

## ∑
η∈Ξ
## |α
ξη
|≤Ch
## 3−2m
## Ξ
, and so Lemma 6.1.2 gives
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (κ
m
## (·,η)− ̃κ
m
## (·,η))
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## ≤
## ∑
η∈Ξ
## |α
ξη
## |‖κ
m
## (·,η)− ̃κ
m
## (·,η)‖
## H
m
≤Ch
## 3−2m
## Ξ
## N
## 2−2m
## .
## (6.2.19)
For the second, start with
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,η)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
## ≤
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |‖ ̃κ
m
## (·,η)‖
## H
m
## .
## Now,‖ ̃κ
m
## (·,η)‖
## H
m
is  bounded  by  a  constant  depending  only  onm.   We
also have
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |=‖α
ξ
## − ̃α
ξ
## ‖
## `
## 1
## (Ξ)
## =
## ∥
## ∥
## ∥
## (
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## )
δ
ξ
## ∥
## ∥
## ∥
## `
## 1
## (Ξ)
## ≤
## ∥
## ∥
## ∥
## K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## .
Now,K
## −1
## Ξ
## −
## ̃
## K
## −1
## Ξ
## =
## ̃
## K
## −1
## Ξ
## (
## ̃
## K
## Ξ
## −K
## Ξ
## )
## K
## −1
## Ξ
, and so by Lemmas 6.2.1, 6.2.2,
and 6.2.3,
## ∑
η∈Ξ
## |α
ξη
## − ̃α
ξη
## |≤
## ∥
## ∥
## ∥
## ̃
## K
## −1
## Ξ
## ∥
## ∥
## ∥
## 1
## ∥
## ∥
## ∥
## ̃
## K
## Ξ
## −K
## Ξ
## ∥
## ∥
## ∥
## 1
## ∥
## ∥
## K
## −1
## Ξ
## ∥
## ∥
## 1
≤Ch
## 3−4m
## Ξ
## N
## 2−2m
## Therefore
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
η∈Ξ
## (α
ξη
## − ̃α
ξη
## ) ̃κ
m
## (·,ζ)
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## H
m
≤Ch
## −4m
## Ξ
## N
## 2−2m
## (6.2.20)
The result now follows by putting (6.2.19) and (6.2.20) into (6.2.7).
## 107

## 6.3    The Truncated Stiffness Matrix
Thetruncated  stiffness  matrixin  the  truncated  Lagrange  basis,
## ̃
## B
## Ξ
,  has
entries
## ̃
## B
ξη
## =〈 ̃χ
ξ
## , ̃χ
η
## 〉
a,b
.  We first establish a bound on the entries of the
difference between the stiffness matrix and the truncated stiffness matrix,
which allows us to get a bound on the 1- and 2-norms of that difference.
Theorem 6.3.1.LetB
## Ξ
and
## ̃
## B
## Ξ
be the stiffness matrix and truncated stiff-
ness matrix,  respectively,  forκ
m
and ̃κ
m
,  respectively,  withm >5/2,  and
centersΞ.   Assume  the  truncation  parameterNsatisfies  the  lower  bound
from  Lemma  6.2.3.   There  is  a  constantCsuch  that,  forh
## Ξ
sufficiently
small,
## ∣
## ∣
## ∣
## B
ξη
## −
## ̃
## B
ξη
## ∣
## ∣
## ∣
≤Ch
## −4m
## Ξ
## N
## 2−2m
## .
Proof.First, we “smuggle in”χ
ξ
## ̃χ
η
and apply H ̈older’s inequality:
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
b(χ
ξ
χ
η
## − ̃χ
ξ
## ̃χ
η
## )dμ
## ∣
## ∣
## ∣
## ∣
## ∣
## ≤b
## 2
## ∫
## SO(3)
## (|χ
ξ
## (χ
η
## − ̃χ
η
## )|
## +|(χ
ξ
## − ̃χ
ξ
## ) ̃χ
η
## |)dμ
## ≤b
## 2
## (
## ‖χ
ξ
## ‖
## L
## 2
## ‖χ
η
## − ̃χ
η
## ‖
## L
## 2
## +‖χ
ξ
## − ̃χ
ξ
## ‖
## L
## 2
## ‖ ̃χ
η
## ‖
## L
## 2
## )
## .
By Lemma 6.2.6, our bump estimate, and Corollary 6.2.4, then,
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
b(χ
ξ
χ
η
## − ̃χ
ξ
## ̃χ
η
## )dμ
## ∣
## ∣
## ∣
## ∣
## ∣
≤Ch
## 9
## 2
## −4m
## Ξ
## N
## 2−2m
## (6.3.2)
We proceed now exactly as we did in the second part of the proof of
Lemma 4.1.5.  CoverSO(3) with finitely many Ω
k
with Ω
k
## ⊆b(q
k
## ,r
## SO(3)
## ),
and letU
k
## = Exp
## −1
k
## (Ω
k
## ).  Let{τ
k
## }
k∈{1,...,K}
be a partition of unity subor-
## 108

dinate to that finite cover.  The exact same arguments give
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
a
## ]
## (∇(χ
ξ
## − ̃χ
ξ
## ),∇(χ
η
## − ̃χ
η
## ))dμ
## ∣
## ∣
## ∣
## ∣
## ∣
## ≤
## ∫
## SO(3)
## ∣
## ∣
a
## ]
## (∇(χ
ξ
## − ̃χ
ξ
## ),∇(χ
η
## − ̃χ
η
## ))
## ∣
## ∣
dμ
## =
## ∥
## ∥
a
## ]
## (∇(χ
ξ
## − ̃χ
ξ
## ),∇(χ
η
## − ̃χ
η
## ))
## ∥
## ∥
## L
## 2
## ≤
## K
## ∑
k=1
## ∑
i,j
## ∥
## ∥
## ∥
## ∥
τ
k
a
ij
## ∂(χ
ξ
## − ̃χ
ξ
## )
## ∂x
i
## ∂(χ
η
## − ̃χ
η
## )
## ∂x
j
## ∥
## ∥
## ∥
## ∥
## L
## 2
## .
Also the exact same arguments give (4.1.8) withχ
ξ
andχ
η
replaced with
χ
ξ
## − ̃χandχ
η
## − ̃χ
η
, respectively.  It thus becomes a matter of bounding
## ∥
## ∥
## ∥
σ
## ∂(χ
ξ
## − ̃χ
ξ
## )
## ∂x
i
## ∥
## ∥
## ∥
## L
## 2
, whereσis the same cutoff function.  Again using the exact
same arguments, we obtain
## ∥
## ∥
## ∥
## ∥
σ
## ∂(χ
ξ
## − ̃χ
ξ
## )
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤C‖χ
ξ
## − ̃χ
ξ
## ‖
## H
## 1
## .
The  Zeros  Lemma  applies  toχ
ξ
## − ̃χ
ξ
,  since  it  vanishes  on  Ξ.   That  and
Lemma 6.2.6 give
## ∥
## ∥
## ∥
## ∥
σ
## ∂(χ
ξ
## − ̃χ
ξ
## )
## ∂x
i
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
m−1
## Ξ
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## H
m
≤Ch
## −3m−1
## Ξ
## N
## 2−2m
## .
Putting this all together gives
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
## (
a
## ]
## (∇χ
xi
## ,∇χ
η
## )−a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
## )
## )
dμ
## ∣
## ∣
## ∣
## ∣
## ∣
≤Ch
## −3m−1
## Ξ
## N
## 2−2m
## ,(6.3.3)
and the result follows from (6.3.2), (6.3.3), and the triangle inequality.
Corollary 6.3.4.Adopt  the  notation  and  assumptions  of  Theorem  6.3.1.
## 109

There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## B
## Ξ
## −
## ̃
## B
## Ξ
## ∥
## ∥
## ∥
## 2
≤Ch
## −3−4m
## Ξ
## N
## 2−2m
## .
Proof.SinceB
## Ξ
and
## ̃
## B
## Ξ
are self-adjoint,
## ∥
## ∥
## ∥
## B
## Ξ
## −
## ̃
## B
## Ξ
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## ∥
## B
## Ξ
## −
## ̃
## B
## Ξ
## ∥
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## B
ξη
## −
## ̃
## B
ξη
## ∣
## ∣
## ∣
## .
For anyξ∈Ξ, we have by Theorem 6.3.1 that
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## B
ξη
## −
## ̃
## B
ξη
## ∣
## ∣
## ∣
≤(#Ξ)·Ch
## −4m
## Ξ
## N
## 2−2m
## ,
and the result follows from #Ξ∼q
## −3
## Ξ
## =ρ
## 3
h
## −3
## Ξ
## .
We need a result similar to the one in Lemma 4.1.25. There we needed to
bound the 1- and 2- norms of the inverse of the quadratized stiffness matrix.
Here we need to bound the 2-norm of the inverse of the truncated stiffness
matrix.  This result requires a lower bound on the truncation parameter,
given in the statement.
Theorem 6.3.5.Adopt  the  notation  and  assumptions  of  Theorem  6.3.1.
LetC
## 0
be the largest constant from Lemma 6.2.1, Lemma 6.2.2, and Corol-
lary 6.3.4.  Assume the truncation parameterNsatisfies
## N≥
## (
## 2C
## 0
h
## −3−4m
## Ξ
## )
## 1
## 2m−3
## .
There is a constantC
## ′
such that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## ≤C
## ′
h
## −3
## Ξ
## .
## 110

Proof.Note that
## ̃
## B
## Ξ
## =B
## Ξ
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## ))
## ,
and hence
## ̃
## B
## −1
## Ξ
## =
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## ))
## −1
## B
## −1
## Ξ
## .
By Theorem 3.4.4 and Corollary 6.3.4,
## ∥
## ∥
## ∥
## B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## )
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## B
## Ξ
## −
## ̃
## B
## Ξ
## ∥
## ∥
## ∥
## 2
≤Ch
## −3−4m
## Ξ
## N
## 3−2m
## .
The assumption onNensures that
## ∥
## ∥
## ∥
## B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## )
## ∥
## ∥
## ∥
## 2
## ≤
## 1
## 2
## ,
which in turn ensures the convergence of the Neumann series
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## ))
## −1
## =
## ∞
## ∑
n=0
## (
## B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## ))
n
## .
## Therefore
## ∥
## ∥
## ∥
## ∥
## (
## I−B
## −1
## Ξ
## (
## B
## Xi
## −
## ̃
## B
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## ∥
## 2
## ≤
## ∞
## ∑
n=0
## ∥
## ∥
## ∥
## B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## )
## ∥
## ∥
## ∥
n
## 2
## ≤
## ∞
## ∑
n=0
## (
## 1
## 2
## )
n
## = 2,
and hence
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## ∥
## ∥
## (
## I−B
## −1
## Ξ
## (
## B
## Ξ
## −
## ̃
## B
## Ξ
## ))
## −1
## ∥
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
## ≤C
## ∥
## ∥
## B
## −1
## Ξ
## ∥
## ∥
## 2
≤Ch
## −3
## Ξ
## .
## 111

## 6.4    Truncated Galerkin Approximation
Thetruncated Galerkin approximationis
## ̃u
## Ξ
## =
## ∑
ξ∈Ξ
## ̃ω
ξ
## ̃χ
ξ
## ,
where the coefficients ̃γ={ ̃γ
ξ
## }
ξ∈Ξ
are obtained by solving the linear system
## ̃
## B
## Ξ
γ= ̃ω, with ̃ω={ ̃ω
ξ
## }
ξ∈Ξ
the vector with entries
## ̃ω
ξ
## =
## ∫
## SO(3)
f ̃χ
ξ
dμ.
Armed with all the results we’ve established thus far, we’re ready to prove
our truncated Galerkin approximation error estimate right away.  As usual,
it will be immediately followed by a result that estimates the error between
the weak solution and the truncated Galerkin approximation - it is just an
application of the triangle inequality.
Theorem 6.4.1.(Truncated Galerkin Approximation Error Estimate)Let
u
## Ξ
and ̃u
## Ξ
be  the  Galerkin  approximation  and  truncated  Galerkin  approx-
imation,  respectively,  forκ
m
and ̃κ
m
,  respectively,  withm >5/2,  centers
Ξ.(Ifκ
m
is  conditionally  positive  definite  with  respect  toΠ
## L
,  assumeΞ
isΠ
## L
-unisolvent.)Assume the truncation parameterNsatisfies the lower
bound in Theorem 6.3.5.  Letf∈H
s
,  wheres=m−1if5/2< m≤7/2
and  lets=m−2ifm >7/2.   There  is  a  constantCsuch  that,  forh
## Ξ
sufficiently small,
## ‖u
## Ξ
## − ̃u
## Ξ
## ‖
## L
## 2
≤Ch
## −
## 15
## 2
## −4m
## Ξ
## N
## 2−2m
## ‖f‖
## H
s
## .
## 112

Remark6.4.2.The reason for the different cases forsis the same as the
one given in Remark 4.2.2, we just haved= 3 now.
Proof.We  start  by  “smuggling  in”
## ∑
ξ∈Ξ
γ
ξ
## ̃χ
ξ
and  applying  the  triangle
inequality:
## ‖u
## Ξ
## − ̃u
## Ξ
## ‖
## L
## 2
## ≤
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
γ
ξ
## (χ
ξ
## − ̃χ
ξ
## )
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## +
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
## (γ− ̃γ) ̃χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
## .(6.4.3)
We handle the summands in (6.4.3) separately.  For the first, we have that,
as in the proof of Lemma 6.2.6, H ̈older’s inequality for sums gives
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
γ
ξ
## (χ
ξ
## − ̃χ
ξ
## )
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
## ≤
## ∑
η∈Ξ
## |γ
η
## |
## 2
## ∑
ζ∈Ξ
## ‖χ
ζ
## − ̃χ
ζ
## ‖
## 2
## L
## 2
## .(6.4.4)
Similarly, for the second summand in (6.4.3),
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
## (γ
ξ
## − ̃γ
ξ
## ) ̃χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
## ≤
## ∑
η∈Ξ
## |γ
ξ
## − ̃γ
ξ
## |
## 2
## ∑
ζ∈Ξ
## ‖ ̃χ
ζ
## ‖
## 2
## L
## 2
## .(6.4.5)
We handle the two factors in (6.4.4) separately.  For the first, we have by
theL
## 2
-stability of the Lagrange basis that
## ∑
η∈Ξ
## |γ
η
## |
## 2
## =‖γ‖
## 2
## `
## 2
## (Ξ)
≤Ch
## −3
## Ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
γ
ξ
χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## 2
## L
## 2
=Ch
## −3
## Ξ
## ‖u
## Ξ
## ‖
## 2
## L
## 2
≤Ch
## −3
## Ξ
## (
## ‖u
## Ξ
## −u‖
## L
## 2
## +‖u‖
## L
## 2
## )
## 2
## .
We’ve  already  seen‖u−u
## Ξ
## ‖
## L
## 2
≤Ch
m−1
## Ξ
## ‖f‖
## H
s
,  and‖u‖
## L
## 2
## ≤ ‖u‖
## H
## 2
## ≤
## 113

## C‖f‖
## L
## 2
≤C‖f‖
## H
s
, and hence
## ∑
η∈Ξ
## |γ
ξ
## |
## 2
≤Ch
## −3
## Ξ
## (
h
m−1
## Ξ
## + 1
## )
## 2
## ‖f‖
## 2
## H
s
≤Ch
## −3
## Ξ
## ‖f‖
## 2
## H
s
## .(6.4.6)
For the second factor in (6.4.4), we have by Theorem 6.2.6 that
## ∑
ζ∈Ξ
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## 2
## L
## 2
≤(#Ξ)·Ch
## 3−8m
## Ξ
## N
## 4−4m
≤Ch
## −8m
## Ξ
## N
## 4−4m
## .
## (6.4.7)
Putting (6.4.6) and (6.4.7) into (6.4.4) and taking square roots, we arrive
at
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
γ
ξ
## (χ
ξ
## − ̃χ
ξ
## )
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
## −
## 3
## 2
## −4m
## Ξ
## N
## 2−2m
## ‖f‖
## H
s
## .(6.4.8)
We now tackle the product in (6.4.5) by handling each factor separately.
For the second factor, we have‖ ̃χ
ξ
## ‖
## 2
## L
## 2
≤C, so
## ∑
ζ∈Ξ
## ‖ ̃χ
ξ
## ‖
## 2
## L
## 2
≤(#Ξ)·C≤Ch
## −3
## Ξ
## .(6.4.9)
For the first factor in (6.4.5), we start by “smuggling in”
## ̃
## B
## −1
## Ξ
ωto obtain
## ∑
η∈Ξ
## |γ
η
## − ̃γ
η
## |
## 2
## =‖γ− ̃γ‖
## 2
## `
## 2
## (Ξ)
## =
## ∥
## ∥
## ∥
## B
## −1
## Ξ
ω−
## ̃
## B
## −1
## Ξ
## ̃ω
## ∥
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## ≤
## (
## ∥
## ∥
## ∥
## (
## B
## −1
## Ξ
## −
## ̃
## B
## −1
## Ξ
## )
ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## (ω− ̃ω)
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## 2
## ≤
## (
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −B
## Ξ
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ∥
## ∥
## ∥
## 2
## ‖ω− ̃ω‖
## `
## 2
## (Ξ)
## )
## 2
## .
As  we  saw  in  the  proof  of  Theorem  4.2.1,
## ∥
## ∥
## B
## −1
## Ξ
ω
## ∥
## ∥
## `
## 2
## (Ξ)
## =‖γ‖
## `
## 2
## (Ξ)
## ≤
## 114

## Ch
## −3/2
## Ξ
## ‖f‖
## H
s
, and Theorems 6.3.4 and 6.3.5 give
## ∑
η∈Ξ
## |γ
η
## − ̃γ
η
## |
## 2
## ≤
## (
## Ch
## −6−4m
## Ξ
## N
## 2−2m
## ‖f‖
## H
s
+Ch
## −3
## Ξ
## ‖ω− ̃ω‖
## `
## 2
## (Ξ)
## )
## 2
## .
## (6.4.10)
We need to deal with the second term in the parentheses in (6.4.10).  We
start by using H ̈older’s inequality to get
## ‖ω− ̃ω‖
## 2
## `
## 2
## (Ξ)
## =
## ∑
ξ∈Ξ
## |ω
ξ
## − ̃ω
ξ
## |
## 2
## =
## ∑
ξ∈Ξ
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
## (χ
ξ
## − ̃χ
ξ
)f dμ
## ∣
## ∣
## ∣
## ∣
## ∣
## 2
## =‖f‖
## 2
## L
## 2
## ∑
ξ∈Ξ
## ‖χ
ξ
## − ̃χ
ξ
## ‖
## 2
## L
## 2
## .
Then (6.4.7) and‖f‖
## L
## 2
## ≤‖f‖
## H
s
give
## ‖ω− ̃ω‖
## 2
## `
## 2
## (Ξ)
≤Ch
## −8m
## Ξ
## N
## 4−4m
## ‖f‖
## 2
## H
m
## .(6.4.11)
Putting the square root of (6.4.11) into (6.4.10) gives
## ∑
η∈Ξ
## |γ
η
## − ̃γ
η
## |
## 2
≤Ch
## −12−8m
## Ξ
## N
## 6−4m
## ‖f‖
## 2
## H
s
## ,(6.4.12)
and putting (6.4.12) and (6.4.9) into (6.4.5) and taking square roots,  we
arrive at
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## ∑
ξ∈Ξ
## (γ
ξ
## − ̃γ
ξ
## ) ̃χ
ξ
## ∥
## ∥
## ∥
## ∥
## ∥
## ∥
## L
## 2
≤Ch
## −
## 15
## 2
## −4m
## Ξ
## N
## 2−2m
## ‖f‖
## H
s
## .(6.4.13)
Finally, putting (6.4.8) and (6.4.13) into (6.4.3) yields the result.
Corollary 6.4.14.Adopt the notation and assumptions of Theorem 6.4.1.
## 115

There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ‖u− ̃u
## Ξ
## ‖
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +h
## −
## 15
## 2
## −4m
## Ξ
## N
## 2−2m
## )
## ‖f‖
## H
s
Remark6.4.15.This brings us to another main point:  we can get the same
result with truncation as we can without by choosing the truncation param-
eterNlarge enough.  We already have the lower bound from Theorem 6.5.3.
If we also haveN≥h
## −
## 5
## 2
## −
## 23
## 4m−4
## Ξ
, then the truncation error,h
## −
## 15
## 2
## −4m
## Ξ
## N
## 2−2m
## ,
will be at most the Galerkin approximation error,h
m−1
## Ξ
## .
## 6.5    The Quadratized Truncated Stiffness Ma-
trix
Thequadratized  truncated  stiffness  matrix,
## ̃
## B
## Λ
## Ξ
## =
## {
## ̃
## B
## Λ
ξη
## }
is  obtained  by
replacing each entry in the truncated stiffness matrix with the quadrature
estimate
## ̃
## B
## Λ
ξη
## =Q
## Λ
## (
a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
## ) +b ̃χ
ξ
## ̃χ
η
## )
## .
Our  first  order  of  business  in  this  section  is  to  bound  the  entries  of  the
difference between the truncated stiffness matrix and the quadratized trun-
cated stiffness matrix.  It will be immediately followed by a bound on the
2-norm of that difference.
Theorem  6.5.1.Let
## ̃
## B
## Ξ
and
## ̃
## B
## Λ
## Ξ
be  the  truncated  stiffness  matrix  and
quadratized stiffness matrix, respectively, for ̃κ
m
withm >5/2, whereΞis
the set of centers used for Galerkin approximation andΛis the set of centers
used  for  quadrature.(Ifκ
m
is  conditionally  positive  definite  with  respect
toΠ
## L
,  assumeΞisΠ
## L
-unisolvent.)Assume#Ξ<#Λandh
## Λ
< h
## Ξ
## .
Suppose the truncation parameterNsatisfies the lower bound in Theorem
## 116

6.3.5.  There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∣
## ∣
## ∣
## ̃
## B
ξη
## −
## ̃
## B
## Λ
ξη
## ∣
## ∣
## ∣
≤Ch
## 2−2m
## Ξ
h
m−1
## Λ
for allξ,η∈Ξ.
Proof.By Theorems 3.1.1 and 6.2.5,
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
b ̃χ
ξ
## ̃χ
η
dμ−Q
## Λ
## (b ̃χ
ξ
## ̃χ
η
## )
## ∣
## ∣
## ∣
## ∣
## ∣
≤Ch
m
## Λ
## ‖b ̃χ
ξ
## ̃χ
η
## ‖
## H
m
≤Ch
m
## Λ
h
## 3−2m
## Ξ
and
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
)dμ−Q
## Λ
## (
a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
## )
## )
## ∣
## ∣
## ∣
## ∣
## ∣
≤Ch
m−1
## Λ
## ∥
## ∥
a
## ]
## (∇ ̃χ
ξ
## ,∇ ̃χ
η
## )
## ∥
## ∥
## H
m−1
≤Ch
m−1
## Λ
h
## 2−2m
## Ξ
## .
Corollary 6.5.2.Adopt the notation and assumptions from Theorem 6.5.1.
There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 2
≤Ch
## −1−2m
## Ξ
h
m−1
## Λ
## .
Proof.This follows the self-adjointness of
## ̃
## B
## Ξ
and
## ̃
## B
## Λ
## Ξ
, Theorem 6.5.1, and
#Ξ∼q
## −3
## Ξ
## =ρ
## 3
h
## −3
## Ξ
## :
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 2
## ≤
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 1
= max
ξ∈Ξ
## ∑
η∈Ξ
## ∣
## ∣
## ∣
## ̃
## B
ξη
## −
## ̃
## B
## Λ
ξη
## ∣
## ∣
## ∣
≤(#Ξ)·Ch
## 2−2m
## Ξ
h
m−1
## Λ
## .
## 117

We end this section with a result regarding the inverse of the quadra-
tized truncated stiffness matrix.  The proof is virtually identical to that of
Theorem 6.3.5.  As such, we omit it.
Theorem 6.5.3.Adopt the notation and assumptions from Theorem 6.5.1.
LetC
## 0
be  the  largest  constant  from  Theorem  6.3.5  and  Corollary  6.5.2.
## Assume
h
## Λ
## ≤C
## 1
m−1
## 0
h
## 2+
## 6
m−1
## Ξ
## 2
## 1
## 1−m
## .
There exists a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ∥
## ∥
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## ∥
## 2
## ≤C
## ′
h
## −3
## Ξ
## .
## 6.6    Quadratized Truncated Approximation
Thequadratized truncated Galerkin approximationis
## ̃u
## Λ
## Ξ
## =
## ∑
ξ∈Ξ
## ̃γ
## Λ
ξ
## ̃χ
ξ
## ,
where  the  coefficients ̃γ
## Λ
## =
## {
## ̃γ
## Λ
ξ
## }
ξ∈Ξ
are  obtained  by  solving  the  linear
system
## ̃
## B
## Λ
## Ξ
## ̃γ
## Λ
## = ̃ω
## Λ
, with ̃ω
## Λ
## =
## {
## ̃ω
## Λ
ξ
## }
ξ∈Ξ
the vector with entries
## ̃ω
## Λ
ξ
## =Q
## Λ
## ( ̃χ
ξ
f).
Armed with the results we have established thus far, we are ready to prove
our  quadratized  truncated  Galerkin  approximation  error  estimate  right
away.   It  will  be  immediately  followed  by  an  error  estimate  between  the
weak solution and the quadratized truncated Galerkin approximation - it
is just repeated applications of the triangle inequality.
## 118

Theorem 6.6.1.(Quadratized Truncated Galerkin Approximation Error
Estimate)Let ̃u
## Ξ
and ̃u
## Λ
## Ξ
be the truncated Galerkin approximation and the
quadratized  truncated  Galerkin  approximation,  respectively,  for ̃κ
m
with
m >5/2,  whereΞis  the  set  of  centers  used  for  Galerkin  approximation
andΛis  the  set  of  centers  used  for  quadrature.   Assume  the  truncation
parameterNsatisfies the the lower bound from Theorem 6.3.5 andh
## Λ
sat-
isfies  both  the  upper  bounds  from  Lemmas  4.1.25  and  6.5.3.  Letf∈H
s
## ,
wheres=m−1if5/2< m≤7/5ands=m−2ifm >7/2.  There is a
constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
## ̃u
## Ξ
## − ̃u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
≤Ch
## −7−2m
## Ξ
h
m−1
## Λ
## ‖f‖
## H
s
## .
Proof.As  in  the  proof  of  Theorem  6.4.1,  we  use  H ̈older’s  inequality  for
sums to obtain
## ∥
## ∥
## ̃u
## Ξ
## − ̃u
## Λ
## Ξ
## ∥
## ∥
## 2
## L
## 2
## ≤
## ∑
η∈Ξ
## ∣
## ∣
## ̃γ
η
## − ̃γ
## Λ
η
## ∣
## ∣
## 2
## ∑
ζ∈Ξ
## ‖ ̃χ
ζ
## ‖
## 2
## L
## 2
## .(6.6.2)
We deal with the factors in (6.6.2) separately.  For the first,  we start by
“smuggling in”
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## ̃ω:
## ∑
η∈Ξ
## ∣
## ∣
## ̃γ
η
## − ̃γ
## Λ
η
## ∣
## ∣
## 2
## =
## ∥
## ∥
## ̃γ− ̃γ
## Λ
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## =
## ∥
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ̃ω−
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## ̃ω
## Λ
## ∥
## ∥
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## ≤
## (
## ∥
## ∥
## ∥
## ∥
## (
## ̃
## B
## −1
## Ξ
## −
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## )
## ̃ω
## ∥
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## ∥
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## (
## ̃ω− ̃ω
## Λ
## )
## ∥
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## 2
## 119

Noting that
## ̃
## B
## −1
## Ξ
## −
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## =
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## (
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## )
## ̃
## B
## −1
## Ξ
, we have
## ∑
η∈Ξ
## ∣
## ∣
## ̃γ
η
## − ̃γ
## Λ
η
## ∣
## ∣
## 2
## ≤
## (
## ∥
## ∥
## ∥
## ∥
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ̃ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ∥
## ∥
## (
## ̃
## B
## Λ
## Ξ
## )
## −1
## ∥
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ̃ω− ̃ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## 2
## Using Theorem 6.5.3,
## ∑
η∈Ξ
## ∣
## ∣
## ̃γ
η
## − ̃γ
## Λ
η
## ∣
## ∣
## 2
≤Ch
## −6
## Ξ
## (
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ̃ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## +
## ∥
## ∥
## ̃ω− ̃ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
## )
## 2
## .
## (6.6.3)
We deal with the terms in the parentheses in (6.6.3) separately.  For the
second, we have by Theorem 3.1.1 withτ=mandσ=s, and 2.7.5,
## ∥
## ∥
## ̃ω− ̃ω
## Λ
## ∥
## ∥
## 2
## `
## 2
## (Ξ)
## =
## ∑
ξ∈Ξ
## (
## ∫
## SO(3)
f ̃χ
ξ
dμ−Q
## Λ
## (f ̃χ
ξ
## )
## )
## 2
≤Ch
## 2m
## Λ
## ∑
ξ∈Ξ
## ‖f ̃χ
ξ
## ‖
## 2
## H
s
≤Ch
## 2m
## Λ
## ∑
ξ∈Ξ
## (
## ‖f‖
## H
s
## ‖ ̃χ
ξ
## ‖
## L
## ∞
## +‖f‖
## L
## ∞
## ‖ ̃χ
ξ
## ‖
## H
s
## )
## 2
## .
As we have already seen,‖f‖
## L
## ∞
≤C‖f‖
## H
s
, and‖ ̃χ
ξ
## ‖
## L
## ∞
≤C‖ ̃χ
ξ
## ‖
## H
s
## ≤
## C‖ ̃χ
ξ
## ‖
## H
m
≤Ch
## 3−2m
## Ξ
.  Hence, using #Ξ∼q
## −3
## Ξ
## =ρ
## 3
h
## −3
## Ξ
and taking square
roots,
## ∥
## ∥
## ̃ω− ̃ω
## Λ
## ∥
## ∥
## `
## 2
## (Ξ)
≤Ch
m
## Λ
h
## 3
## 2
## −2m
## Ξ
## ‖f‖
## H
s
## .(6.6.4)
For the first term in the parentheses in (6.6.3), note first that
## ∥
## ∥
## ∥
## ̃
## B
## −1
## ̃ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
## ≤
## ∥
## ∥
## ∥
## ̃
## B
## −1
## ∥
## ∥
## ∥
## 2
## ‖ ̃ω‖
## `
## 2
## (Ξ)
≤Ch
## −3
## Ξ
## ‖ ̃ω‖
## `
## 2
## (Ξ)
## .
Using H ̈older’s inequality, #Ξ∼h
## −3
## Ξ
yet again, and the fact that‖ ̃χ
ξ
## ‖
## L
## 2
## ≤
## 120

## C,
## ‖ ̃ω‖
## 2
## `
## 2
## (Ξ)
## =
## ∑
ξ∈Ξ
## ∣
## ∣
## ∣
## ∣
## ∣
## ∫
## SO(3)
f ̃χ
ξ
dμ
## ∣
## ∣
## ∣
## ∣
## ∣
## 2
## ≤‖f‖
## 2
## L
## 2
## ∑
ξ∈Ξ
## ‖ ̃χ
ξ
## ‖
## 2
## L
## 2
≤Ch
## −3
## Ξ
## ‖f‖
## 2
## L
## 2
## .
## Therefore,
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ̃ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
≤Ch
## −9/2
## Ξ
## ‖f‖
## L
## 2
≤Ch
## −9/2
## Ξ
## ‖f‖
## H
s
, and so by Corol-
lary 6.5.2
## ∥
## ∥
## ∥
## ̃
## B
## Ξ
## −
## ̃
## B
## Λ
## Ξ
## ∥
## ∥
## ∥
## 2
## ∥
## ∥
## ∥
## ̃
## B
## −1
## Ξ
## ̃ω
## ∥
## ∥
## ∥
## `
## 2
## (Ξ)
≤Ch
## −
## 11
## 2
## −2m
## Ξ
h
m−1
## Λ
## ‖f‖
## H
s
## .(6.6.5)
Putting (6.6.4) and (6.6.5) into (6.6.3), we arrive at
## ∑
η∈Ξ
## ∣
## ∣
## ̃γ
η
## − ̃γ
## Λ
η
## ∣
## ∣
## 2
≤Ch
## −17−4m
## Ξ
h
## 2m−2
## Λ
## ‖f‖
## 2
## H
s
## .(6.6.6)
For the second factor in (6.6.2), we have by Corollary 6.2.4 and yet again
#Ξ∼h
## −3
## Ξ
that
## ∑
ζ∈Ξ
## ‖ ̃χ
ζ
## ‖
## 2
## L
## 2
≤Ch
## 3−4m
## Ξ
## (6.6.7)
The result now follows by putting (6.6.6) and (6.6.7) into (6.6.2) and taking
square roots.
Corollary 6.6.8.Adopt the notation and assumptions from Theorem 6.6.1.
There is a constantCsuch that, forh
## Ξ
sufficiently small,
## ∥
## ∥
u− ̃u
## Λ
## Ξ
## ∥
## ∥
## L
## 2
## ≤C
## (
h
m−1
## Ξ
## +h
## −
## 15
## 2
## −4m
## Ξ
## N
## 2−2m
## +h
## −7−2m
## Ξ
h
m−1
## Λ
## )
## ‖f‖
## H
s
## .
Remark6.6.9.We  come  now  to  our  last  main  point:  we  can  do  as  well
with quadrature as we can without ifh
## Λ
is small enough.  In particular, if
## 121

h
## Λ
## ≤h
p
## Ξ
with oversampling exponentp= 3 +
## 9
m−1
, then the quadratization
error,h
## −7−2m
## Ξ
h
m−1
## Λ
will be exactly the Galerkin approximation error,h
m−1
## Ξ
(of course higher oversampling exponents will do even better).
We also must keep in mind the upper bound forh
## Λ
from Lemma 4.1.25,
which  is  likely  more  restrictive  than  the  bound  given  byh
## Λ
## ≤h
p
## Ξ
## .   On
SO(3) we haved= 3, and so the bound is
h
## Λ
## ≤C
## 4
## 5−2m
## 0
h
## 1+
## 16
## 2m−5
## Ξ
## 2
## 5
## 2
## −m
## .
If indeed the quadrature weights satisfy the lower bound discussed in Re-
mark 4.1.26, then that upper bound is not needed, and the error estimates
in Corollary 6.6.8 alone tell us how to chooseh
## Λ
## .
## 122

## Bibliography
[1]  N. Aronszajn. La th ́eorie des noyaux reproduisants et ses applications.
I.Proc. Cambridge Philos. Soc., 39:133–153, 1943.
[2]  N.  Aronszajn.   Theory  of  reproducing  kernels.Trans.  Amer.  Math.
## Soc., 68:337–404, 1950.
[3]  T. Aubin.  Espaces de Sobolev sur les vari ́et ́es Riemanniennes.Bull.
## Sci. Math. (2), 100(2):149–173, 1976.
[4]  M. D. Buhmann. Multivariate cardinal interpolation with radial-basis
functions.Constr. Approx., 6(3):225–255, 1990.
[5]  T. Coulhon, E. Russ, and V. Tardivel-Nachef. Sobolev algebras on Lie
groups and Riemannian manifolds.Amer.  J.  Math., 123(2):283–342,
## 2001.
[6]  M.  P.  do  Carmo.Differential  geometry  of  curves  and  surfaces.
Prentice-Hall, Inc., Englewood Cliffs, N.J., 1976.  Translated from the
## Portuguese.
[7]  M.  P.  do  Carmo.Riemannian  geometry.   Mathematics:   Theory  &
Applications. Birkh ̈auser Boston, Inc., Boston, MA, 1992.  Translated
from the second Portuguese edition by Francis Flaherty.
## 123

[8]  J. Duchon. Sur l’erreur d’interpolation des fonctions de plusieurs vari-
ables  par  lesD
m
-splines.RAIRO  Anal.  Num ́er.,  12(4):325–334,  vi,
## 1978.
[9]  G. B. Folland.Introduction to partial differential equations. Mathemat-
ical Notes. Princeton University Press, Princeton, N.J., 1976. Prelimi-
nary informal notes of university courses and seminars in mathematics.
[10]  G. B. Folland.A course in abstract harmonic analysis.  Textbooks in
Mathematics. CRC Press, Boca Raton, FL, second edition, 2016.
[11]  E. Fuselier, T. Hangelbroek, F. J. Narcowich, J. D. Ward, and G. B.
Wright. Localized bases for kernel spaces on the unit sphere.SIAM J.
## Numer. Anal., 51(5):2538–2562, 2013.
[12]  E. Fuselier, T. Hangelbroek, F. J. Narcowich, J. D. Ward, and G. B.
Wright.  Kernel based quadrature on spheres and other homogeneous
spaces.Numer. Math., 127(1):57–92, 2014.
[13]  A.  Gulisashvili  and  M.  A.  Kon.Exact  smoothing  properties  of
Schr ̈odinger semigroups.Amer. J. Math., 118(6):1215–1248, 1996.
[14]  T.  Hangelbroek,  F.  J.  Narcowich,  C.  Rieger,  and  J.  D.  Ward.   Di-
rect  and  inverse  results  on  bounded  domains  for  meshless  methods
via  localized  bases  on  manifolds.    InContemporary  computational
mathematics—a  celebration  of  the  80th  birthday  of  Ian  Sloan.  Vol.
1, 2, pages 517–543. Springer, Cham, 2018.
[15]  T. Hangelbroek, F. J. Narcowich, and J. D. Ward. Kernel approxima-
tion on manifolds I: bounding the Lebesgue constant.SIAM J. Math.
## Anal., 42(4):1732–1760, 2010.
## 124

[16]  T. Hangelbroek, F. J. Narcowich, and J. D. Ward.  Polyharmonic and
related kernels on manifolds: interpolation and approximation.Found.
## Comput. Math., 12(5):625–670, 2012.
[17]  T.  Hangelbroek  and  D.  Schmid.    Surface  spline  approximation  on
SO(3).Appl. Comput. Harmon. Anal., 31(2):169–184, 2011.
[18]  S. Helgason.Groups and geometric analysis, volume 83 ofMathemat-
ical Surveys and Monographs.  American Mathematical Society, Prov-
idence,  RI,  2000.   Integral  geometry,  invariant  differential  operators,
and spherical functions, Corrected reprint of the 1984 original.
[19]  A.  Hoorfar  and  M.  Hassani.   Inequalities  on  the  LambertWfunc-
tion and hyperpower function.JIPAM. J. Inequal. Pure Appl. Math.,
9(2):Article 51, 5, 2008.
[20]  R. C. Hoover, A. A. Maciejewski, and R. G. Roberts.  Pose detection
of  3-d  objects  using  images  sampled  on  so(3),  spherical  harmonics,
and  wigner-d  matrices.   In2008  IEEE  International  Conference  on
Automation Science and Engineering, pages 47–52, 2008.
[21]  W. R. Madych and S. A. Nelson.  Polyharmonic cardinal splines.J.
## Approx. Theory, 60(2):141–156, 1990.
[22]  F. J. Narcowich, S. T. Rowe, and J. D. Ward. A novel Galerkin method
for  solving  PDEs  on  the  sphere  using  highly  localized  kernel  bases.
## Math. Comp., 86(303):197–231, 2017.
[23]  F. J. Narcowich, J. D. Ward, and H. Wendland.  Sobolev bounds on
functions with scattered zeros, with applications to radial basis func-
tion surface fitting.Math. Comp., 74(250):743–763, 2005.
## 125

[24]  F. Riesz and B. Sz.-Nagy.Functional analysis.  Frederick Ungar Pub-
lishing Co., New York, 1955.  Translated by Leo F. Boron.
[25]  R.  Schaback.   Native  Hilbert  spaces  for  radial  basis  functions.  I.   In
New developments in approximation theory (Dortmund, 1998), volume
132 ofInternat. Ser. Numer. Math., pages 255–282. Birkh ̈auser, Basel,
## 1999.
[26]  D. Schmid.Scattered Data Approximation on the Rotation Group and
Generalizations.  Berichte aus der Mathematik. Shaker, 2009.
[27]  R. S. Strichartz. Analysis of the Laplacian on the complete Riemannian
manifold.J. Functional Analysis, 52(1):48–79, 1983.
[28]  H. Triebel. Spaces of Besov-Hardy-Sobolev type on complete Rieman-
nian manifolds.Ark. Mat., 24(2):299–337, 1986.
[29]  H. Wendland.Scattered data approximation, volume 17 ofCambridge
Monographs on Applied and Computational Mathematics.  Cambridge
## University Press, Cambridge, 2005.
## 126

ProQuest Number:
## INFORMATION TO ALL USERS
The quality and completeness of this reproduction is dependent on the quality
and completeness of the copy made available to ProQuest.
Distributed by ProQuest LLC (        ).
Copyright of the Dissertation is held by the Author unless otherwise noted.
This work may be used in accordance with the terms of the Creative Commons license
or other rights statement, as indicated in the copyright statement or in the metadata
associated with this work. Unless otherwise specified in the copyright statement
or the metadata, all rights are reserved by the copyright holder.
This work is protected against unauthorized copying under Title 17,
United States Code and other applicable copyright laws.
Microform Edition where available © ProQuest LLC. No reproduction or digitization
of the Microform Edition is authorized without permission of ProQuest LLC.
ProQuest LLC
## 789 East Eisenhower Parkway
P.O. Box 1346
Ann Arbor, MI 48106 - 1346 USA
## 28717615
## 2021