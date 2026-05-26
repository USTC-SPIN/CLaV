import { PAPER } from '../data/site'

export default function Abstract() {
  return (
    <section id="abstract" className="anchor-offset border-b border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-14 lg:py-16">
        <div className="eyebrow">Abstract</div>
        <p className="mt-5 font-display text-[18px] leading-[1.65] text-ink/90 text-justify hyphens-auto" lang="en">
          {PAPER.abstract}
        </p>
      </div>
    </section>
  )
}
