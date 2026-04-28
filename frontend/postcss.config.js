// PostCSS pipeline.
//
// Tailwind v4 ships its own PostCSS plugin (`@tailwindcss/postcss`) that
// replaces the v3 `tailwindcss` plugin. Autoprefixer still runs after it so
// non-Tailwind CSS in `index.scss` (e.g. `-webkit-line-clamp` fallbacks,
// scrollbar pseudo-elements) gets vendor prefixes for the browserslist
// defaults.
export default {
  plugins: {
    '@tailwindcss/postcss': {},
    autoprefixer: {},
  },
}
