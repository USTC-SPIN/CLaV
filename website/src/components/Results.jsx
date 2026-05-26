import { useState } from 'react'
import { RESULTS, ABLATION } from '../data/site'
import { asset } from '../lib/asset'

function fmt(x) {
  if (x == null) return '—'
  return x.toFixed(2)
}

const TABS = [
  ['table', 'Numbers'],
  ['ablation', 'Ablation'],
  ['radar', 'Radar'],
  ['tsne', 't-SNE'],
]

export default function Results() {
  const [tab, setTab] = useState('table')
  return (
    <section id="results" className="anchor-offset border-b border-rule">
      <div className="mx-auto max-w-[920px] xl:max-w-[1080px] 2xl:max-w-[1200px] px-6 lg:px-10 py-14 lg:py-16">
        <div className="eyebrow">Results</div>
        <h2 className="mt-3 font-display text-2xl lg:text-[32px] leading-[1.15] text-ink">
          State-of-the-art on rain and snow across every dataset.
        </h2>
        <p className="mt-3 font-display text-[16px] text-ink/85 leading-[1.7] text-justify hyphens-auto" lang="en">
          Under a like-for-like denoise-then-retrieve protocol, BEV U-Net signal-space denoising reaches
          <span className="num"> 37.00%</span> KITTI R@1 — our latent flow matching reaches
          <span className="num"> 62.84%</span>.
          <span className="ml-1 font-display italic text-[14px] text-graphite">R@1 / R@5 with a 3 m positive radius.</span>
        </p>

        <div className="mt-8 inline-flex border border-rule">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-3.5 py-1.5 font-display text-[14px] border-r last:border-r-0 border-rule transition-colors ${
                tab === id ? 'bg-ink text-paper' : 'bg-paper text-graphite hover:text-ink'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {tab === 'table' && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full font-display text-[14px] border border-rule">
              <thead>
                <tr>
                  {['Dataset', 'Weather', 'R@1 prev', 'R@5 prev', 'R@1 ours', 'R@5 ours'].map((h, i) => (
                    <th
                      key={h}
                      className={`p-2.5 font-display italic text-[12.5px] text-graphite font-normal border-b border-rule ${i < 2 ? 'text-left' : 'text-right'}`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {RESULTS.flatMap((d) =>
                  d.weathers.map((w, i) => (
                    <tr key={d.dataset + w.w} className="border-t border-rule">
                      {i === 0 && (
                        <td rowSpan={d.weathers.length} className="align-top p-2.5 font-display text-[16px] text-ink">
                          {d.dataset}
                        </td>
                      )}
                      <td className="p-2.5 text-graphite">{w.w}</td>
                      <td className="p-2.5 text-right num text-graphite">{fmt(w.prevR1)}</td>
                      <td className="p-2.5 text-right num text-graphite">{fmt(w.prevR5)}</td>
                      <td className="p-2.5 text-right num font-medium text-ink">{fmt(w.r1)}</td>
                      <td className="p-2.5 text-right num font-medium text-ink">{fmt(w.r5)}</td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
            <p className="mt-2.5 font-display italic text-[12.5px] text-graphite">
              Prev = best published baseline per row. Numbers in %.
            </p>
          </div>
        )}

        {tab === 'ablation' && (
          <div className="mt-5 overflow-x-auto">
            <table className="w-full font-display text-[14px] border border-rule">
              <thead>
                <tr>
                  {['Configuration', 'KITTI R@1', 'KITTI R@5', 'NCLT R@1', 'NCLT R@5'].map((h, i) => (
                    <th
                      key={h}
                      className={`p-2.5 font-display italic text-[12.5px] text-graphite font-normal border-b border-rule ${i === 0 ? 'text-left' : 'text-right'}`}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ABLATION.map((r) => (
                  <tr key={r.row} className="border-t border-rule">
                    <td className={`p-2.5 ${r.best ? 'text-ink font-medium' : 'text-graphite'}`}>{r.row}</td>
                    <td className={`p-2.5 text-right num ${r.best ? 'text-ink font-medium' : 'text-graphite'}`}>{fmt(r.k_r1)}</td>
                    <td className={`p-2.5 text-right num ${r.best ? 'text-ink font-medium' : 'text-graphite'}`}>{fmt(r.k_r5)}</td>
                    <td className={`p-2.5 text-right num ${r.best ? 'text-ink font-medium' : 'text-graphite'}`}>{fmt(r.n_r1)}</td>
                    <td className={`p-2.5 text-right num ${r.best ? 'text-ink font-medium' : 'text-graphite'}`}>{fmt(r.n_r5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2.5 font-display italic text-[12.5px] text-graphite">Ablation (B) — paper Tab. 3.</p>
          </div>
        )}

        {tab === 'radar' && (
          <figure className="mt-5">
            <div className="border border-rule bg-paper">
              <img src={asset('assets/paper/recall5.png')} alt="Radar plot of Recall@5 across three datasets and three weathers." className="w-full h-auto block" loading="lazy" />
            </div>
            <figcaption className="mt-2 font-display italic text-[13px] text-graphite leading-snug">
              <span className="label-ink not-italic mr-2">Fig. 2</span>
              Recall@5 across KITTI · NCLT · Boreas under rain · fog · snow.
            </figcaption>
          </figure>
        )}

        {tab === 'tsne' && (
          <figure className="mt-5">
            <div className="border border-rule bg-paper">
              <img src={asset('assets/paper/tsne.png')} alt="Joint t-SNE projection before and after latent denoising." className="w-full h-auto block" loading="lazy" />
            </div>
            <figcaption className="mt-2 font-display italic text-[13px] text-graphite leading-snug">
              <span className="label-ink not-italic mr-2">Fig. 3</span>
              Joint t-SNE before / after denoising.
            </figcaption>
          </figure>
        )}
      </div>
    </section>
  )
}
