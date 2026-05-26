import { useState } from 'react'
import { RETRIEVAL_CASES } from '../data/site'
import { asset } from '../lib/asset'

export default function RetrievalDemo() {
  const [idx, setIdx] = useState(0)
  const c = RETRIEVAL_CASES[idx]
  return (
    <section id="cases" className="anchor-offset border-b border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-14 lg:py-16">
        <div className="eyebrow">Cases</div>
        <h2 className="mt-3 font-display text-2xl lg:text-[28px] leading-[1.2] text-ink">
          Query → Top-5 retrieval, mapped on the database trajectory.
        </h2>
        <p className="mt-3 font-display text-[16px] text-ink/85 leading-[1.7] text-justify hyphens-auto" lang="en">
          Each case shows the query BEV, the database trajectory (positives in green), a zoomed 50 m local view,
          and the five retrieved frames in the order returned by the descriptor.
        </p>

        <div className="mt-7 flex flex-wrap gap-1.5">
          {RETRIEVAL_CASES.map((rc, i) => (
            <button
              key={rc.id}
              onClick={() => setIdx(i)}
              className={`px-3.5 py-1.5 border font-display text-[14px] transition-colors ${
                i === idx ? 'bg-ink text-paper border-ink' : 'bg-paper text-graphite border-rule hover:text-ink hover:border-ink'
              }`}
            >
              {rc.title}
            </button>
          ))}
        </div>

        <figure className="mt-5">
          <div className="border border-rule bg-paper">
            <img src={asset(c.src)} alt={c.title} className="w-full h-auto block" />
          </div>
          <figcaption className="mt-3 font-display text-[14px] text-graphite leading-snug italic">
            <span className="label-ink not-italic mr-2">{c.title}</span>
            {c.summary}
          </figcaption>
        </figure>
      </div>
    </section>
  )
}
