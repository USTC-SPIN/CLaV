// Build a public asset URL that respects Vite's `base` (so it works both
// in dev (base = /) and on GitHub Pages (base = /clav/)).
export const asset = (path) => {
  const base = import.meta.env.BASE_URL || '/'
  const clean = path.replace(/^\/+/, '')
  return `${base}${clean}`
}
