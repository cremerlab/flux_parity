---
layout: page
title: Code and Data
permalink: code
sidebar: true
---

---
## Code

All code used to perform the analysis and generate the figures used in this work
will be made public upon submission of the paper to the bioRxiv preprint server. 

## Data
All data shown this work will be hosted here upon submission of the paper to a 
preprint server.

<!-- {% if site.data.code %} -->

<!-- {% for script in site.data.code %}
* [**{{script.name}}**]({{site.url}}/{{site.baseurl}}/software/{{script.name}})
  \| {{script.desc}}
{% endfor %}
{% endif %}

{% if site.data.datasets %}
## Data Sets
{% for ds in site.data.datasets %}
* [{{ds.name}}]({%if ds.storage !=
  'remote'%}{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}{%
  else%}{{ds.link}}{% endif %}) \| {% if ds.filetype %}(filetype:
  {{ds.filetype}}){%endif%}{% if ds.filesize %}({{ds.filesize}}){%endif%}{%
  if ds.storage == remote %} DOI: {{ds.DOI}}{%endif%}
{% endfor %}
{% endif %}

{% if site.data.figures %}
## Figure Generation

{% for fig in site.data.figures %}
<article class="post">

<a class="post-thumbnail" style="background-image: url({{site.url}}/{{site.baseurl}}/assets/img/{{fig.pic}})" href="{{site.baseurl}}/figures/{{fig.pdf}}"> </a>

<div class="post-content">
<b class="post-title"><a href="{{site.url}}/{{site.baseurl}}/software/{{fig.filename}}">{{fig.title}}</a></b>
<p> {{fig.desc}}</p>

<i>Necessary Data Sets </i><br/>
{% for ds in fig.req %}
{% if ds.storage == 'local' %}
{% assign link = "{{site.url}}/{{site.baseurl}}/datasets/{{ds.link}}" %}
{% else %}
{% assign link = "{{ds.link}}" %}
{% endif %}
<a style="font-size: 0.9em;" href="{{link}}"> - {{ds.title}} </a><br/>
{% endfor %}
</div>
</article>
{%endfor%}
{% endif %} -->