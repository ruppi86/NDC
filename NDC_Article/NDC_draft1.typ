Här är en **utbyggbar** artikel-outline som matchar exakt vad ni har byggt (claims-matrix, A1/A2, provenance, negative controls, MNJ vs ΔY vs random). Jag skriver den så att ni enkelt kan stoppa in kommande “nästa golden/dataset”-delar utan att riva strukturen.

---

## Titel (arbetsnamn)

**Benchmarking Jacobian-Gated Segmentation for Nonstationary Neurodynamic Signals**
(alt: *A Claims-First Benchmark for Gating and Regime Segmentation using Local Jacobians*)

---

## Abstract

1–2 meningar om problemet: icke-stationära tidsserier/regimskiften, behov av robust segmentering som inte bara är amplitudkänslig.
1–2 meningar om ansatsen: claims-first benchmark med separata within-regime och cross-regime tester, plus negativa kontroller och provenance.
1–2 meningar om resultat: MNJ-gating visar icke-trivial effekt (slår random med marginal i A1), men är ännu inte bättre än enklare ΔY-gating på Golden A; tail-stress (A2) visar förväntad generaliseringskollaps.
Avslut: release av kod/benchmark och roadmap mot fler dataset.

---

## 1. Introduction

* Motivera varför regimskiften är centrala (neurodynamik, kognition, kliniska tillstånd, generellt nonstationära system).
* Problem: många metoder blir antingen “för fria” (tunar sig till dataset) eller “för otestade” (saknar negativa kontroller och falsifierbarhet).
* Bidrag: ni introducerar ett benchmarkramverk där varje claim har test + negativ kontroll + förväntat failure mode, samt en tydlig separation mellan within-regime och cross-regime generalisering.

**Utbyggbarhet:** sista stycket kan senare nämna “vi utökar med fler goldens och riktiga dataset i vNext”.

---

## 2. Related Work

* Kort om change-point detection och piecewise modeller (klassiska metoder, men håll det kort).
* Kort om gradient/amplitud-baserade “gates” (ΔY-liknande heuristiker) och varför de ofta är starka men kan missledas.
* Lokal linjär approximation/Jacobian-idéer (lokal dynamik, tangent space, identifiability) som intuition för MNJ.

**Utbyggbarhet:** här kan ni senare lägga till jämförelser mot specifika state-of-the-art när ni väl väljer dem.

---

## 3. Problem Setup and Notation

* Definiera observerad signal (Y(t)), latent/embedded representation (MNPS) och målet: hitta en split (s) som förbättrar prediktion under piecewise modell.
* Klargör skillnad mellan:

  * *split selection* (var placerar vi skiftet?)
  * *evaluation* (hur mäter vi förbättring?)
* Definiera “prediction gain” som primär metrik: förbättring relativt persistence-baseline.

**Utbyggbarhet:** här kan ni senare lägga in fler segment (2+ change points) som framtida.

---

## 4. Methods

### 4.1 MNPS embedding

* Beskriv MNPS som en lågdimensionell representation med fast (k) (t.ex. 3) och normalisering (z-score), med diagnostik för stabilitet.
* Motivera varför embed behövs innan Jacobian-estimering (redundans/noise i råa dimensioner).

### 4.2 MNJ: Local Jacobian estimation (v0/v1)

* Lokal regression på grannar i MNPS-rummet för att estimera (J(t)) (ridge på Jacobian-del, ej intercept).
* Tydliggör att MNJ-resultatet innehåller residualer och konditionsmått för “identifiability”.

### 4.3 Gating signals

* ΔY-gate: (g_{\Delta Y}(t)=|Y_{t}-Y_{t-1}|).
* MNJ-gate: (g_{MNJ}(t)=|J(t)|_F) ev viktad av trust-score (om ni behåller Policy B som valbar; annars beskriv i appendix).

### 4.4 Split selection policy (viktig!)

* Beskriv policyförändringen som gjorde ramen tydligare:

  * tidigare: piecewise SSE-fit på gatesignalen (kan vara oavsiktligt)
  * nu: **gate→split = argmax(gate) inom giltig domän** (semantiskt switch locator)
* Förklara varför argmax är “claim-aligned”: om gate är “switch-energien”, då är maximum en naturlig kandidat.

**Utbyggbarhet:** här kan ni senare lägga till “soft-argmax / centroid” som framtida, men inte i mainline.

---

## 5. Benchmark Design (Claims-first)

### 5.1 Claims matrix

* Förklara att varje claim har: test, negativ kontroll, expected failure mode.
* Referera till er claims.yaml som “living spec”.

### 5.2 Golden A: piecewise regime switch

* Beskriv generativ setup på hög nivå (switch vid (t_{switch}), oracle finns).
* Förklara varför Golden A är “minimal men skarp”: den ger en känd förändringspunkt och låter er testa identifiability vs heuristik.

### 5.3 A1 vs A2: honest_middle och honest_tail

* A1 (within-regime): eval i mittenfönster, blandad regim men inte ren tail → huvudtest för segmentering.
* A2 (cross-regime stress): tail-only eval → stress-test som kan gå negativt och ska rapporteras som warn, inte fail.

### 5.4 Negative controls och sanity checks

* Shuffled collapse: om man shuffle:ar ska gating/segmentation kollapsa.
* Random split baseline: gate ska slå random med marginal för att räknas som icke-trivial.
* Provenance-fält: split idx/time, |idx–oracle|, signal stats, objective name.

**Utbyggbarhet:** ett underavsnitt kan senare bli “Additional Goldens (decoy amplitude spikes, gradual drift, multi-switch)”.

---

## 6. Experiments

* Lista exakt vilka benchmark som körs (Golden A1/A2, Golden B2, invariance stress).
* Ange fixed policies (min_seg_seconds, domain, selection_fraction, oracle_opt_window_seconds, margin=0.01).
* Ange att inga “nya DOF” introduceras utan explicit policybeslut.

**Utbyggbarhet:** här kan ni senare lägga in “real datasets: ds000115 …” med samma tabellformat.

---

## 7. Results

### 7.1 Golden A1 (primary)

* Rapportera tabellen med gains: random-split, ΔY-gate, MNJ-gate, och MNJ-shuffled.
* Huvudobservation: MNJ-gate **slår random med marginal** under argmax-split, men **slår inte ΔY** (än).
* Tolka: MNJ är “signalbärande” men inte dominerande i detta golden-läge.

### 7.2 Golden A2 (stress)

* Rapportera global honest_tail-negativitet som förväntad drift/generaliseringskollaps.
* Förklara varför A2 är warn: den säger mer om dataset/regimdrift än om gating-bugg.

### 7.3 Provenance: single source of truth

* Visa att gating-signaler är icke-degenererade (nonzero_fraction, mean/std).
* Visa split-lokalisering: gate-argmax idx och avstånd till oracle.
* Visa att negative controls beter sig rätt (shuffled collapse).

### 7.4 Golden B2 + invariance stress (kort)

* B2 pass: (kort, visar att pipeline håller).
* invariance_stress warn: (kort, beskriver vad det betyder och att det inte blockerar A1-claimen).

**Utbyggbarhet:** här kan ni senare lägga in “MNJ wins on decoy golden” som egen subsubsection om ni får det.

---

## 8. Discussion

* Varför ΔY kan vara svårt att slå: amplitude proxy är stark på enkla switchar.
* Varför MNJ ändå är intressant: fångar struktur i lokal dynamik, potentiellt mer robust mot vissa decoys (hypotes).
* Vad trust_coverage betyder vs trust_score: identifiability är hårt, gating kan vara mjukt och ändå användbart—men måste rapporteras transparent (vilket ni nu gör).

**Utbyggbarhet:** här kan ni lägga in “predictions” om vilka dataset/regimer där MNJ borde vinna.

---

## 9. Limitations

* Endast en primär golden för MNJ-v1 (än).
* MNJ underpresterar mot ΔY i Golden A; resultatet är därför “proof of signal” snarare än “best method”.
* trust_coverage låg: identifiability-kriteriet är strikt och bör tolkas korrekt (inte förväxlas med gating-degenerering).
* Argmax-split är en policy: rimlig men inte “enda sanna”—dock explicit och claim-aligned.

---

## 10. Roadmap / Future Work (det här gör outline “utbyggbar”)

* **Golden: amplitude-spike decoy** där ΔY luras men MNJ förväntas hålla sig närmare oracle.
* **Golden: gradual drift** (ingen hård switch) för att se om MNJ signalerar “instabilitet” snarare än punkt.
* **Real datasets**: ett eller två OpenNeuro-dataset med fenomenologi/metadata.
* Multi-switch (2 change points) och robust split-strategi (men behåll claims-first).

---

## 11. Reproducibility and Release

* GitHub repo + Zenodo DOI.
* Manifest-driven benchmark, rapportgenerering MD/JSON, claims-matrix som “spec”.
* Exakta versionspinnar (report_version, manifest_version).

---

## Appendix

### A. Benchmark Manifest & Claims YAML (kort excerpt)

### B. Metric definitions (prediction gain, random baseline, shuffled collapse)

### C. MNJ diagnostics fields (rel_mse_base, cond, excitation, effective rank)

### D. Policy log (en rad per policyförändring med motivation)

---

Om du vill kan jag också:

* skriva ett “Abstract v1” i er stil (tight, reviewer-friendly), eller
* föreslå 3–4 figuridéer som matchar exakt ert report-format (och som är enkla att generera från JSON).
