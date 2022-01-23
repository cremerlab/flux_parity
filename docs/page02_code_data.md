---
layout: page
title: Data
permalink: data
sidebar: true
img: seg.png
---

---
## Code
All of the code used in the analysis of this work is available on the [GitHub repository associated with this work](https://github.com/cremerlab/flux_parity) along with instructions regarding installation of the custom Python module

## Data
All of the data used in the analysis of this work is available on the [GitHub repository associated with this work](https://github.com/cremerlab/flux_parity). Below, we provide links to individual data sets associated with key figures.

{% if site.data.datasets %}
{% for ds in site.data.datasets %}
* [{{ds.name}}]({%if ds.storage !=
  'remote'%}{{site.baseurl}}/datasets/{{ds.link}}{%
  else%}{{ds.link}}{% endif %}) \| {% if ds.filetype %}(filetype:
  {{ds.filetype}}){%endif%}{% if ds.filesize %}({{ds.filesize}}){%endif%}{%
  if ds.storage == remote %} DOI: {{ds.DOI}}{%endif%}
{% endfor %}
{% endif %}