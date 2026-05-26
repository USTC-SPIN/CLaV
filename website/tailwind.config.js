/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        paper: '#ffffff',     // plain white background
        ink: '#1a1f2b',       // deep slate ink
        graphite: '#3b424f',  // body-secondary
        petrol: '#0e5468',    // primary accent — radar/instrument blue
        petrolHi: '#1e7a90',  // hover/lighter accent
        rust: '#a85a2a',      // metric highlight
        rustHi: '#c2723f',
        rule: '#d8d8d8',      // neutral hairline rules
        ruleSoft: '#ececec',
        chrome: '#2b303a',    // dark mono chrome blocks
      },
      fontFamily: {
        display: ['"Fraunces"', 'ui-serif', 'Georgia', 'serif'],
        body: ['"Fraunces"', 'ui-serif', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      maxWidth: {
        prose: '68ch',
        measure: '74ch',
      },
      keyframes: {
        sweep: {
          '0%':   { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
        riseIn: {
          '0%':   { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        sweep: 'sweep 5s linear infinite',
        riseIn: 'riseIn 0.7s ease-out both',
      },
    },
  },
  plugins: [],
}
