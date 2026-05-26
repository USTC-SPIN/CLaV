import { useState } from 'react'
import { BIBTEX } from '../data/site'

export default function Bibtex() {
  const [copied, setCopied] = useState(false)
  return (
    <section id="bibtex" className="anchor-offset">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-14 lg:py-16">
        <div className="eyebrow">Cite</div>
        <div className="mt-5 relative">
          <pre className="code-block whitespace-pre overflow-x-auto">{BIBTEX}</pre>
          <button
            onClick={() => {
              navigator.clipboard.writeText(BIBTEX)
              setCopied(true)
              setTimeout(() => setCopied(false), 1200)
            }}
            className="absolute top-2 right-2 font-display text-[12px] text-paper/80 hover:text-paper border border-paper/20 px-2 py-0.5"
          >
            {copied ? 'copied' : 'copy'}
          </button>
        </div>
      </div>
    </section>
  )
}
