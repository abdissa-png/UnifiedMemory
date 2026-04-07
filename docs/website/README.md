# Docusaurus site (MDX deep dive)

This package renders the chapters under [`../mdx/`](../mdx/) with **Mermaid** diagram support.

```bash
npm install
npm start          # http://localhost:3000
npm run build      # static site → build/
npm run serve      # preview production build
```

Requires **Node.js ≥ 18**.

Configuration: `docusaurus.config.js`, `sidebars.js`. Content path: `path: '../mdx'` (relative to this directory).
