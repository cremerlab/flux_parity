---
layout: page
title: Interactive Scenarios
permalink: scenarios
sidebar: true
interactive: interactive_strategies.html
---
---

## Exploring Allocation Scenarios
In the main text of the paper, we outline three plausible strategies cells may 
employ to determine the ribosomal allocation. These hypotheses are:

I. **Allocation is constant across conditions (color: black).** This  means that the cell is 
"locked" into a specific physiology and will always maintain the proteomic composition, 
regardless of the environment. 

II. **Allocation is tuned to maintain elongation rate (color:  green).** Under this scenario, 
the ribosomal allocation is tuned such that the precursor concentration is maintained
at the same level across conditions. This results in a constant translation rate 
across conditions.

II. **Allocation is optimally tuned to achieve the fastest growth rate (color: blue).** Here, 
the ribosomal content is tuned such that the growth rate in a given environmental 
condition *and a given allocation to other proteins* is maximized. This is not 
the same as saying growth rate is maximized,  as is discussed in the main text. 

For all three scenarios, we've derived closed-form expressions defining the 
ribosomal allocation parameter, which is presented in the main text. Here, 
we allow you to explore how different parameter combinations can change the 
behavior of these scenarios when examining the ribosomal allocation, translation rate, 
and steady-state growth rate. 

Note that scenario I (constant allocation) allows you to pick whatever ribosomal 
allocation parameter you would like. This is done by adjusting the slider in the 
bottom-right corner of the control panel. Adjusting this slider does not impact 
the behavior of the other scenarios. 

<!-- The below line includes the interactive figure. Do not change! -->
<center>

{% include_relative interactives/{{page.interactive}} %}

</center>


