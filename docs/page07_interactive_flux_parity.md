---
layout: page
title: Interactive Flux-Parity
permalink: tents
sidebar: true
interactive: interactive_equilibration.html
---
---

## Exploring the Flux-Parity Model
In the supplemental text, we explore how at an intuitive level how flux-parity 
regulation results in an optimal allocation of ribosomes. Here, we provide an 
interactive version of Fig. S7. 

In the figure below, we have computed the how allocation under the 
flux-parity model influences growth rate, shown as a black line. Any point 
on this line satisfies a steady-state condition, but not necessarily the optimal. 
At any point on this curve (shown as a black point), one can compute the metabolic 
flux $$\nu(tRNA)(1 - \phi_O - \phi_{Rb})$$ and the 
translational flux $$\gamma(tRNA^*)\phi_{Rb}$$ to see that they are equal (overlap), 
but have different slopes. The optimal allocation is the point at which any 
perturbation to $$\phi_{Rb}$$ results in a decrease in *both* fluxes. At this 
point, the fluxes are mutually maximized. 

<!-- The below line includes the interactive figure. Do not change! -->
<center>

{% include_relative interactives/{{page.interactive}} %}

</center>


