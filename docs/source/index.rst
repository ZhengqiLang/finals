Documentation for the logRASM artifact
=====================

This repository contains the supplementary code for the paper:

*`Policy Verification in Stochastic Dynamical Systems Using Logarithmic Neural Certificates<https://doi.org/10.48550/arXiv.2406.00826>`_ by Thom Badings, Wietze Koops, Sebastian
Junges, and Nils Jansen (CAV 2025)*

This paper proposes techniques that make the verification of neural network policies in stochastic dynamical systems more scalable.
In this artifact, we implement these techniques in a learner-verifier framework for verifying that a given neural network policy satisfies a given reach-avoid specification.
The learner trains another neural network, which acts as a certificate proving that the policy satisfies the task.
The verifier then checks whether this neural network certificate is a so-called logarithmic reach-avoid supermartingale (logRASM), which suffices to show reach-avoid guarantees.
For more details about the approach, we refer to the paper above.

Source code on GitHub: https://github.com/LAVA-LAB/logRASM


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ReadMe
   modules