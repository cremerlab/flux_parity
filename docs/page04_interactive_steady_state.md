---
layout: page
title: Interactive Steady-State
permalink: steady_state
sidebar: true
interactive: interactive_steadystate.html
---
---

## Exploring the closed-form steady-state behavior
In the main text of the paper, we discuss that inclusion of the dilution in our 
expression for the precursor concentration allows us to calculate analytical 
expressions for a variety of steady-state properties of the ribosomal allocation 
model. Their complete derivations are provided in the Supplementary Information 
but we state their forms here. The steady-state precursor concentration is defined 
as 
$$
c_{pc}^* = \frac{\mathsf{N}}{\lambda} - 1, \tag{1} 
$$
where $\mathsf{N}$  is the maxium metabolic output, $\mathsf{N} = \nu_{max}(1 - \phi_{Rb} - \phi_O)$.
With knowledge of the precursor concentration, we can then easily calculate the 
steady-state translation rate 
$$
\gamma(c_{pc}^*) = \gamma_{max}\frac{c_{pc}^*}{c_{pc}^* + K_D^{c_{pc}}} \tag{2}.
$$

With these in hand, we can then calculate an expression for the steady-state 
growth rate, which is a quadratic equation with only one physically meaningful root 
of 
$$
\lambda = \frac{\mathsf{N} + \Gamma - \sqrt{(\mathsf{N} + \Gamma)^2 - 4\mathsf{N}\Gamma(1 - K_D^{c_{pc})}}{2(1 - K_D^{c_{pc}})}, \tag{3}
$$
were we have introduce the notation of $\Gamma = \gamma_{max}\phi_{Rb}$, which 
is the maximum translational output. 

A property evident in these expressions is that they all depend on the ribosomal 
allocation parameter, $\phi_{Rb}$. In the figure below, we plot the behavior 
of these three expressions as a function of the ribosomal allocation $\phi_{Rb}$,
and allow the user to tune the remaining model parameters to gauge the importance 
of each parameter. The parameter $\nu_{max}$ can be thought of as a proxy for the 
quality of the nutrients in the environment with low values (light green in the figure) 
corresponding to poor environments and large values (dark blue in the figure) corresponding
to rich conditions.


<!-- The below line includes the interactive figure. Do not change! -->
<center>

{% include_relative interactives/{{page.interactive}} %}

</center>


