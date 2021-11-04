---
layout: page
title: Interactive Data Comparison
permalink: scenarios
sidebar: true
interactive: interactive_ecoli_data_comparison.html
---
---

## Comparing Allocation Scenarios to *E. coli* Data 
In the main text of the paper, we outline three plausible strategies cells may 
employ to determine the ribosomal allocation. These hypotheses are:

I. **Allocation is constant across conditions (color: black).** 

II. **Allocation is tuned to maintain elongation rate (color:  green).** 

II. **Allocation is optimally tuned to achieve the fastest growth rate (color: blue).** 

You can learn more about these scenarios and their parametric sensitivity on 
the [Interactive Scenarios]({{site.baseurl}}/scenarios) page of this website. 
Here, we compare the predictions from these different scenarios to experimental 
measurements of ribosomal allocation and transalation rate as a function of the 
growth rate in the model microbe *Escherichia coli*. The data in this figure 
comes from a variety of sources and methods. You can hover over each point to 
identify the source, data values, and the method of measurement. References to 
each are given at the bottom of this page.   

Interaction with the sliders changes the parameters of the three scenarios. 
Note that scenario I (constant allocation) allows you to pick whatever ribosomal 
allocation parameter you would like. This is done by adjusting the slider in the 
bottom-right corner of the control panel. Adjusting this slider does not impact 
the behavior of the other scenarios. 

<!-- The below line includes the interactive figure. Do not change! -->
<center>

{% include_relative interactives/{{page.interactive}} %}

</center>


