import { PIPELINE } from '../data/site'
import { asset } from '../lib/asset'

export default function Method() {
  return (
    <section id="method" className="anchor-offset border-b border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-14 lg:py-16">
        <div className="eyebrow">Method</div>
        <h2 className="mt-3 font-display text-2xl lg:text-[28px] leading-[1.2] text-ink">
          A frozen semantic latent, then a velocity field that transports noise into clean retrieval features.
        </h2>
        <div className="mt-2 font-display italic text-[14px] text-graphite">
          Ω = h ∘ ψ ∘ E ∘ φ
        </div>

        <figure className="mt-8">
          <div className="border border-rule bg-paper">
            <img
              src={asset('assets/paper/architecture.png')}
              alt="C-LaV full architecture diagram."
              className="w-full h-auto block"
              loading="lazy"
            />
          </div>
          <figcaption className="mt-2 font-display italic text-[13px] text-graphite leading-snug">
            <span className="label-ink not-italic mr-2">Fig. 1</span>
            The full C-LaV pipeline. At inference, only the noisy branch is active: φ → E → ψ → h.
          </figcaption>
        </figure>

        <ol className="mt-10 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 lg:gap-2">
          {PIPELINE.map((p, i) => (
            <li key={p.sym} className="relative">
              <div className="border border-rule bg-paper p-4 h-full">
                <div className="flex items-baseline justify-between">
                  <span className="font-display text-[34px] leading-none text-ink">{p.sym}</span>
                  <span className="font-display italic text-[12px] text-graphite">{i + 1} / 4</span>
                </div>
                <div className="mt-3 font-display text-[15.5px] text-ink leading-tight">{p.name}</div>
                <div className="mt-2 font-mono text-[11.5px] text-graphite">
                  <span className="text-ink">{p.chipLabel}</span>
                  <span className="mx-1.5 text-rule">·</span>
                  <span>{p.chipShape}</span>
                </div>
              </div>
              {i < PIPELINE.length - 1 && (
                <span
                  aria-hidden
                  className="hidden lg:flex absolute top-1/2 -right-2 -translate-y-1/2 font-display text-graphite text-lg z-10 bg-paper px-1"
                >
                  →
                </span>
              )}
            </li>
          ))}
        </ol>

        <p className="mt-7 font-display text-[16px] text-ink/85 leading-[1.7] text-justify hyphens-auto" lang="en">
          A single-sweep point cloud is projected to a three-channel BEV, encoded by a frozen
          DINOv2-Base into a 32×32 latent grid, denoised by an ODE-integrated velocity field, and
          aggregated by SALAD into a compact 8 448-dim descriptor used directly for retrieval.
        </p>
      </div>
    </section>
  )
}
