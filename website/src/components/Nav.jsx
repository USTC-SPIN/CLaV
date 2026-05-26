import { useEffect, useState } from 'react'

const ITEMS = [
  { id: 'abstract', label: 'Abstract' },
  { id: 'method', label: 'Method' },
  { id: 'results', label: 'Results' },
  { id: 'cases', label: 'Cases' },
  { id: 'bibtex', label: 'BibTeX' },
]

export default function Nav() {
  const [active, setActive] = useState('abstract')
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 4)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })

    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-45% 0px -50% 0px', threshold: 0 },
    )
    ITEMS.forEach(({ id }) => {
      const el = document.getElementById(id)
      if (el) obs.observe(el)
    })
    return () => {
      window.removeEventListener('scroll', onScroll)
      obs.disconnect()
    }
  }, [])

  return (
    <nav
      className={`sticky top-0 z-40 bg-paper/90 backdrop-blur border-b transition-colors ${
        scrolled ? 'border-rule' : 'border-transparent'
      }`}
    >
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 h-14 flex items-center gap-8">
        <a href="#top" className="shrink-0 font-display text-[15px] text-ink hover:text-petrol transition-colors">
          C-LaV <span className="text-graphite">·</span> CVPR 2026
        </a>
        <ul className="hidden lg:flex items-center gap-6 ml-2 flex-1">
          {ITEMS.map((it) => (
            <li key={it.id}>
              <a
                href={`#${it.id}`}
                className={`font-display text-[15px] transition-colors ${
                  active === it.id ? 'text-petrol' : 'text-graphite hover:text-ink'
                }`}
              >
                {it.label}
              </a>
            </li>
          ))}
        </ul>
        <a
          className="ml-auto font-display text-[15px] text-graphite hover:text-ink"
          href="https://github.com/Patience-Joey/clav"
          target="_blank"
          rel="noreferrer"
        >
          GitHub ↗
        </a>
      </div>
    </nav>
  )
}
