---
title: 'idinn: A Python Package for Inventory-Dynamics Control with Neural Networks'
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
Despite over 60 years of extensive research on inventory management problems, even fundamental 
dual-sourcing problems [@barankin1961delivery,@fukuda1964optimal]—where orders from an 
expensive supplier arrive faster than orders from a regular supplier—remain analytically 
intractable. Additionally, there is a growing interest in optimization algorithms that 
are capable of handling real-world inventory problems with large numbers of 
suppliers and non-stationary demand.

We provide a Python package, `idinn`, implementing inventory dynamics–informed neural 
networks designed for controlling both single-sourcing and dual-sourcing problems. 
Neural network controllers and inventory dynamics are implemented in two easily customizable 
classes, enabling users to control extensions of the provided inventory management 
systems by tailoring the implementations to their needs. `idinn` also encompasses 
a dynamic program that computes the optimal solution to dual-sourcing problems. 

# Statement of need

Inventory management problems commonly arise in almost all industries. A basic and 
yet analytically intractable problem in inventory management is dual sourcing 
[@barankin1961delivery,@fukuda1964optimal]. `idinn` is a Python package for controlling 
dual-sourcing inventory dynamics with dynamics-informed neural networks. 
Unlike traditional reinforcement-learning approaches, our optimization approach takes 
into account how the system being optimized behaves over time, leading to more efficient training 
and accurate solutions. 

Training neural networks for inventory-dynamics control presents 
a specific challenge. The adjustment of neural network weights during training relies 
on propagating real-valued gradients, whereas the neural network outputs—representing 
replenishment orders—must be integers. To address this challenge in optimizing a 
discrete problem with real-valued gradient descent learning algorithms, we employ 
a problem-tailored straight-through estimator [@yang2022injecting,@asikis2023multi]. 
This approach enables us to obtain integer-valued neural network outputs while 
backpropagating real-valued gradients.

`idinn` has been developed for researchers and students working at the intersection 
of optimization, operations research, and machine learning. It has been made available 
to students in a machine learning course at Frankfurt School to demonstrate 
the effectiveness of artificial neural networks in solving real-world optimization problems.
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