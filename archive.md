---
layout: default
title: Archive
permalink: /archive/
---

<h1 style="text-align: center;">Archive</h1>

<div class="archive">
  {% assign postsByYear = site.posts | group_by_exp: "post", "post.date | date: '%Y'" %}
  
  {% if postsByYear.size > 0 %}
    <ul class="archive-list">
      {% for year in postsByYear %}
        <li>
          <h2 class="archive-year">{{ year.name }}</h2>
          <ul class="archive-list">
            {% for post in year.items %}
              <li class="archive-item">
                <span class="archive-date">{{ post.date | date: "%b %-d" }}</span>
                <a href="{{ post.url | relative_url }}">{{ post.title | escape }}</a>
              </li>
            {% endfor %}
          </ul>
        </li>
      {% endfor %}
    </ul>
  {% else %}
    <p class="no-posts">No posts yet. Check back soon!</p>
  {% endif %}
</div>