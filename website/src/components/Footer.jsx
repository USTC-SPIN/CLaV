export default function Footer() {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-8 flex flex-col md:flex-row gap-3 md:items-baseline md:justify-between">
        <div className="font-display text-[11.5px] text-graphite italic whitespace-nowrap overflow-hidden text-ellipsis min-w-0">
          University of Science and Technology of China · School of Artificial Intelligence and Data Science · Suzhou Institute for Advanced Research
        </div>
        <div className="flex items-center gap-5 font-display text-[12px] text-graphite shrink-0">
          <a className="hover:text-ink" href="https://github.com/Patience-Joey/clav" target="_blank" rel="noreferrer">Code ↗</a>
          <a className="hover:text-ink" href="#top">Top</a>
          <span className="italic">© 2026</span>
        </div>
      </div>
    </footer>
  )
}
