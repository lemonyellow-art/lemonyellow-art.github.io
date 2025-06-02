# Art Portfolio Website

A modern, minimalist Jekyll-based portfolio website for showcasing artwork and illustrations.

## Features

- Clean, responsive three-column gallery layout
- Tag-based artwork organization
- Modern design with smooth animations
- Mobile-friendly interface

## Setup

1. Install Jekyll and Ruby dependencies:
```bash
gem install bundler
bundle install
```

2. Start the development server:
```bash
bundle exec jekyll serve
```

3. Visit `http://localhost:4000` to view your site

## Adding New Artwork

1. Create a new `.md` file in the `_artwork` directory
2. Use the following front matter template:
```yaml
---
title: "Artwork Title"
date: YYYY-MM-DD
image: /assets/images/your-image.jpg
tags: 
  - tag1
  - tag2
description: "Brief description of the artwork"
---
```

3. Add your artwork image to the `assets/images` directory
4. Write your artwork description using Markdown below the front matter

## Directory Structure

- `_artwork/`: Individual artwork post files
- `assets/images/`: Artwork images
- `assets/css/`: Stylesheets
- `_layouts/`: Template files
- `tags.html`: Tag-based gallery page
- `index.html`: Homepage with gallery grid

## Customization

- Edit `_config.yml` to change site settings
- Modify `assets/css/main.css` to customize styles
- Update `_layouts/default.html` to change the site structure