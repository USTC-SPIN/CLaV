// Static content for the C-LaV project page.
// Numbers and quotes are sourced from the CVPR 2026 paper.

export const PAPER = {
  title: 'C-LaV: Conditional Latent Velocity Field Denoising',
  subtitle: 'Weather-Robust LiDAR Place Recognition',
  venue: 'CVPR 2026',
  authors: [
    { name: 'Xuewei Cao',  aff: '1,2', mail: 'caoxuewei@mail.ustc.edu.cn' },
    { name: 'Jiayue Yang', aff: '1,2', mail: 'jiayueyang@mail.ustc.edu.cn' },
    { name: 'Zhiwen Zeng', aff: '1,2', mail: '20220665@stu.cqu.edu.cn' },
    { name: 'Yanyong Zhang', aff: '1,2', mail: 'yanyongz@ustc.edu.cn' },
    { name: 'Yan Xia',     aff: '1,2', corresponding: true, mail: 'yan.xia@ustc.edu.cn' },
  ],
  affiliations: [
    'School of Artificial Intelligence and Data Science, University of Science and Technology of China',
    'Suzhou Institute for Advanced Research, University of Science and Technology of China',
  ],
  abstract:
    'LiDAR-based place recognition is highly sensitive to rain, snow, and fog, where scattering and attenuation distort geometric structure and intensity. We tackle this problem with Conditional Latent Velocity Field (C-LaV) denoising, which restores weather-robust representations before retrieval. Single-sweep point clouds are projected into three-channel bird’s-eye-view (BEV) images and encoded with a frozen DINOv2-based BEV transformer to obtain a semantically anchored latent space shared across weather conditions. On this manifold, a conditional Flow-Matching model learns a velocity field whose probability-flow ODE deterministically transports noisy latents toward their clear-weather counterparts. From the denoised manifold, a Sinkhorn Aggregation of Local Descriptors (SALAD) head produces compact global descriptors optimized with a truncated Smooth-AP loss. We also establish a unified adverse-weather benchmark with 3 m frame spacing and shared evaluation thresholds across KITTI, NCLT, and Boreas. Our C-LaV improves Recall@1 by 17.5% on NCLT snow and 21.5% on Boreas, achieving state-of-the-art weather robustness.',
  links: {
    paper: 'https://openaccess.thecvf.com/content/CVPR2026/html/Cao_C-LaV_Conditional_Latent_Velocity_Field_Denoising_for_Weather-Robust_LiDAR_Place_CVPR_2026_paper.html',
    code:  'https://github.com/USTC-SPIN/CLaV',
    weights: 'https://huggingface.co/xueweicao/clav',
    bibtex: '#bibtex',
  },
}

export const PIPELINE = [
  {
    sym: 'φ',
    name: 'BEV projection',
    body: 'Single-sweep point cloud is rasterised onto a 448×448 BEV grid, with three channels encoding normalised height, mean reflectance, and clipped point density.',
    chipLabel: 'I',
    chipShape: '448 × 448 × 3',
  },
  {
    sym: 'E',
    name: 'Frozen DINOv2 encoder',
    body: 'A frozen DINOv2-Base (ViT/14) maps the BEV to a 32×32 latent grid of 768-dim tokens — a semantically anchored manifold shared across weather.',
    chipLabel: 'Z₀',
    chipShape: '768 × 32 × 32',
  },
  {
    sym: 'ψ',
    name: 'Conditional velocity field',
    body: 'A Flow-Matching velocity vθ(z, t; cond) is integrated by an ODE that deterministically transports noisy latents toward their clear-weather counterparts.',
    chipLabel: 'Z_d ≈ Z_clean',
    chipShape: '768 × 32 × 32',
  },
  {
    sym: 'h',
    name: 'SALAD aggregation',
    body: 'Sinkhorn-Aggregation of Local Descriptors collapses 1024 tokens into a compact ℓ₂-normalised global descriptor of 8448 dimensions, used directly for retrieval.',
    chipLabel: 'D',
    chipShape: 'ℝ⁸⁴⁴⁸',
  },
]

// Recall@1 / Recall@5 per dataset × weather. Bolded numbers below are also the figure highlights.
export const RESULTS = [
  {
    dataset: 'KITTI',
    weathers: [
      { w: 'Rain', r1: 46.97, r5: 67.88, prevR1: 46.16, prevR5: 65.15, prev: 'MinkLoc3D v2' },
      { w: 'Fog',  r1: 62.73, r5: 85.45, prevR1: 67.07, prevR5: 87.78, prev: 'MinkLoc3D v2' },
      { w: 'Snow', r1: 77.60, r5: 95.23, prevR1: 72.28, prevR5: 90.31, prev: 'ImLPR' },
    ],
  },
  {
    dataset: 'NCLT',
    weathers: [
      { w: 'Rain', r1: 29.49, r5: 55.69, prevR1: 28.61, prevR5: 51.42, prev: 'MinkLoc3D v2' },
      { w: 'Fog',  r1: 28.87, r5: 56.58, prevR1: 29.81, prevR5: 57.91, prev: 'MinkLoc3D v2' },
      { w: 'Snow', r1: 46.41, r5: 69.11, prevR1: 30.81, prevR5: 51.62, prev: 'MinkLoc3D v2' },
    ],
  },
  {
    dataset: 'Boreas',
    weathers: [
      { w: 'Rain', r1: 79.66, r5: 98.31, prevR1: 65.52, prevR5: 86.32, prev: 'MinkLoc3D v2' },
      { w: 'Snow', r1: 71.98, r5: 94.83, prevR1: 52.37, prevR5: 81.98, prev: 'BEVPlace++ / ImLPR' },
    ],
  },
]

// Ablation B from the paper.
export const ABLATION = [
  { row: 'C-LaV-1 · DINOv2-S + DDPM + NetVLAD',        k_r1: 11.17, k_r5: 30.20, n_r1:  5.40, n_r5: 17.85 },
  { row: 'C-LaV-2 · DINOv2-B + DDPM + NetVLAD',        k_r1: 30.45, k_r5: 51.20, n_r1: 16.80, n_r5: 38.55 },
  { row: 'C-LaV-3 · DINOv2-B + Velocity + NetVLAD',    k_r1: 50.15, k_r5: 71.90, n_r1: 27.35, n_r5: 53.80 },
  { row: 'C-LaV · DINOv2-B + Velocity + SALAD (ours)', k_r1: 62.83, k_r5: 82.75, n_r1: 34.52, n_r5: 60.16, best: true },
]

export const RETRIEVAL_CASES = [
  {
    id: 'kitti-rain',
    title: 'KITTI · Sequence 02 · Rain',
    src: 'assets/demo/retrieval/kitti_rain.png',
    summary:
      'A rainy single sweep is encoded, denoised, and queried against the sunny database of the same sequence. Top-5 are all within the 3 m positive threshold (drawn in green).',
    chips: ['rain', 'sequence 02', 'top-5 ✓ 5/5'],
  },
  {
    id: 'kitti-fog',
    title: 'KITTI · Sequence 02 · Fog',
    src: 'assets/demo/retrieval/kitti_fog.png',
    summary:
      'Fog attenuates far-range structure. Even so, the denoised descriptor recovers a tightly-clustered Top-5 around the query location.',
    chips: ['fog', 'sequence 02', 'top-5 ✓ 5/5'],
  },
  {
    id: 'nclt-snow',
    title: 'NCLT · Snow',
    src: 'assets/demo/retrieval/nclt_snow.png',
    summary:
      'NCLT covers a different sensor platform and broader environments. Even under snow, retrieved frames sit within a few metres of the ground-truth pose.',
    chips: ['snow', 'NCLT', 'cross-platform'],
  },
]

export const BIBTEX = `@inproceedings{cao2026clav,
  title     = {C-LaV: Conditional Latent Velocity Field Denoising for Weather-Robust LiDAR Place Recognition},
  author    = {Cao, Xuewei and Yang, Jiayue and Zeng, Zhiwen and Zhang, Yanyong and Xia, Yan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}`
