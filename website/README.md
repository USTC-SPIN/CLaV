# C-LaV Project Page

Static project page for **C-LaV** (CVPR 2026) — deployed to GitHub Pages at
<https://patience-joey.github.io/clav/> via the workflow in `.github/workflows/deploy.yml`.

## Layout

```
website/
├── index.html
├── src/
│   ├── App.jsx                  # section composition
│   ├── components/              # Nav / Hero / Abstract / Method / Results / RetrievalDemo / Bibtex / Footer
│   ├── data/site.js             # content: abstract, pipeline, results table, ablation, bibtex
│   └── lib/asset.js             # base-aware asset URLs
└── public/assets/
    ├── paper/                   # teaser, architecture, t-SNE, radar
    ├── demo/retrieval/          # 3 retrieval trace cases
    └── video/                   # demo.mp4 + poster
```
