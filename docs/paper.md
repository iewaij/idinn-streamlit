---
title: 'IDINN: A Python package for inventory-dynamics control with neural networks'
tags:
  - Python
  - artificial neural networks
  - inventory dynamics
  - optimization
  - control
  - dynamic programming
authors:
  - name: Jiawei Li
    affiliation: 1
    corresponding: true
  - name: Thomas Asikis
    orcid: 0000-0003-0163-4622
    affiliation: 2
  - name: Ioannis Fragkos
    affiliation: 3
  - name: Lucas B\"ottcher
    affiliation: "1,4"
    orcid: 0000-0003-1700-1897
affiliations:
 - name: Department of Computational Science and Philosophy, Frankfurt School of Finance and Management
   index: 1
 - name: Game Theory, University of Zurich
   index: 2
 - name: Department of Technology and Operations Management, Rotterdam School of Management, Erasmus University Rotterdam
   index: 3
 - name: Laboratory for Systems Medicine, Department of Medicine, University of Florida
   index: 4
date: 16 January 2024
bibliography: paper.bib

---

# Summary

Identifying optimal policies for replenishing inventory from multiple suppliers is a key 
problem in inventory management. Solving such optimization problems means that one must 
determine the quantities to order from each supplier based on the current net inventory 
and outstanding orders, minimizing the expected backlogging, holding, and sourcing costs. 
Despite over 60 years of extensive study on inventory management problems, even fundamental 
dual-sourcing problems—where orders from an expensive supplier arrive faster than orders 
from a regular supplier—remain analytically intractable. Additionally, there is a growing 
need for optimization algorithms that are capable of handling real-world inventory 
problems with large numbers of suppliers and non-stationary demand.

We provide a Python package, `IDINN`, implementing inventory dynamics–informed neural 
networks designed for controlling dual-sourcing problems. We also provide an implementation 
of a dynamic program that computes the optimal solution to dual-sourcing problems. 
The package includes neural-network architecture data and dynamic-program outputs 
for 72 dual-sourcing instances, serving as a valuable testbed for evaluating dual-sourcing 
optimization algorithms.

# Statement of need

`Gala` is an Astropy-affiliated Python package for galactic dynamics. Python
enables wrapping low-level languages (e.g., C) for speed without losing
flexibility or ease-of-use in the user-interface. The API for `Gala` was
designed to provide a class-based and user-friendly interface to fast (C or
Cython-optimized) implementations of common operations such as gravitational
potential and force evaluation, orbit integration, dynamical transformations,
and chaos indicators for nonlinear dynamics. `Gala` also relies heavily on and
interfaces well with the implementations of physical units and astronomical
coordinate systems in the `Astropy` package [@astropy] (`astropy.units` and
`astropy.coordinates`).

`IDINN` has been developed for researchers and students working at the intersection 
of optimization, operations research, and machine learning. It has been made available 
to students in a machine learning course at Frankfurt School to demonstrate 
the effective application of artificial neural networks in solving real-world optimization problems.
In a previous publication [@bottcher2023control], a less accessible code base was used to
compute near-optimal solutions of dozens of dual-sourcing instances. 

# Brief software description

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Conflicts of interest

The authors declare that they have no conflicts of interest.

# References