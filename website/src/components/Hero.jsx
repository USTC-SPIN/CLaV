import { PAPER } from '../data/site'
import { asset } from '../lib/asset'

export default function Hero() {
  return (
    <header id="top" className="anchor-offset border-b border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 pt-12 lg:pt-16 pb-14">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-8 font-display text-[14.5px] text-graphite italic">
          <span className="text-petrol not-italic">{PAPER.venue}</span>
          <span>·</span>
          <span>LiDAR place recognition under adverse weather</span>
        </div>

        <h1 className="font-display font-medium leading-[1.1] tracking-tight text-ink text-[28px] sm:text-[32px] lg:text-[38px] [text-wrap:balance]">
          C-LaV: Conditional Latent Velocity Field Denoising for Weather-Robust LiDAR Place Recognition
        </h1>

        <div className="mt-8 max-w-3xl">
          <div className="flex flex-wrap gap-x-5 gap-y-1.5 font-display text-[16px] text-ink/90">
            {PAPER.authors.map((a) => (
              <span key={a.name} className="whitespace-nowrap">
                {a.name}
                <sup className="text-[11px] ml-0.5 text-graphite">{a.aff}</sup>
                {a.corresponding && <sup className="text-graphite"> †</sup>}
              </span>
            ))}
          </div>
          <ol className="mt-3 list-none p-0 text-[13.5px] text-graphite font-display italic">
            {PAPER.affiliations.map((aff, i) => (
              <li key={i}>
                <sup className="text-[11px] mr-1 not-italic">{i + 1}</sup>
                {aff}
              </li>
            ))}
            <li className="mt-1"><sup className="text-[11px] mr-1 not-italic">†</sup>Corresponding author</li>
          </ol>
        </div>

        <div className="mt-7 flex flex-wrap gap-2">
          <a className="btn btn-primary" href={PAPER.links.paper} target="_blank" rel="noreferrer">
            <span>Paper</span><span aria-hidden>↗</span>
          </a>
          <a className="btn" href={PAPER.links.code} target="_blank" rel="noreferrer">
            <span>Code</span><span aria-hidden>↗</span>
          </a>
          <a className="btn" href={PAPER.links.weights} target="_blank" rel="noreferrer">
            <span>Weights</span><span aria-hidden>↗</span>
          </a>
          <a className="btn" href="#cases">
            <span>Cases</span><span aria-hidden>↓</span>
          </a>
          <a className="btn" href="#bibtex">
            <span>BibTeX</span>
          </a>
        </div>

        <div className="mt-12 border border-rule bg-ink">
          <video
            controls
            preload="metadata"
            poster={asset('assets/video/demo_poster.jpg')}
            className="w-full h-auto block bg-ink"
          >
            <source src={asset('assets/video/demo.mp4')} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </header>
  )
}
