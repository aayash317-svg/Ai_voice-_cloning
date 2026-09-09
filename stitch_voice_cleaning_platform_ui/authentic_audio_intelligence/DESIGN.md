---
name: Authentic Audio Intelligence
colors:
  surface: '#111317'
  surface-dim: '#111317'
  surface-bright: '#37393d'
  surface-container-lowest: '#0c0e11'
  surface-container-low: '#1a1c1f'
  surface-container: '#1e2023'
  surface-container-high: '#282a2d'
  surface-container-highest: '#333538'
  on-surface: '#e2e2e6'
  on-surface-variant: '#bbcabf'
  inverse-surface: '#e2e2e6'
  inverse-on-surface: '#2f3034'
  outline: '#86948a'
  outline-variant: '#3c4a42'
  surface-tint: '#4edea3'
  primary: '#4edea3'
  on-primary: '#003824'
  primary-container: '#10b981'
  on-primary-container: '#00422b'
  inverse-primary: '#006c49'
  secondary: '#4cd7f6'
  on-secondary: '#003640'
  secondary-container: '#03b5d3'
  on-secondary-container: '#00424e'
  tertiary: '#ffb3ad'
  on-tertiary: '#68000a'
  tertiary-container: '#ff7a73'
  on-tertiary-container: '#79000e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#6ffbbe'
  primary-fixed-dim: '#4edea3'
  on-primary-fixed: '#002113'
  on-primary-fixed-variant: '#005236'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#ffdad7'
  tertiary-fixed-dim: '#ffb3ad'
  on-tertiary-fixed: '#410004'
  on-tertiary-fixed-variant: '#930013'
  background: '#111317'
  on-background: '#e2e2e6'
  surface-variant: '#333538'
typography:
  display:
    fontFamily: Inter
    fontSize: 40px
    fontWeight: '600'
    lineHeight: 48px
    letterSpacing: -0.03em
  display-mobile:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 38px
    letterSpacing: -0.025em
  headline-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '500'
    lineHeight: 28px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 24px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-code:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-badge:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.04em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  space-2xs: 0.25rem
  space-xs: 0.5rem
  space-sm: 0.75rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
  space-2xl: 3rem
  space-3xl: 4rem
  gutter-mobile: 1rem
  gutter-desktop: 1.5rem
  nav-bottom-height: 4.5rem
---

## Brand & Style

This design system embodies the calculated precision of high-tier forensic intelligence paired with the restrained elegance of luxury consumer security hardware. The product operates at the intersection of critical verification and effortless usability; its tone is authoritative, unshakeable, and transparent.

The aesthetic merges **Dark Luxury** with **Minimal Glassmorphism**:
- Deep void-like backdrops eliminate distraction, focusing attention entirely on acoustic telemetry, spectrogram fidelity, and probabilistic risk metrics.
- Surfaces are treated as machined lenses—frosted obsidian layers framed by microscopic light leaks (subtle translucent borders) that signal architectural integrity.
- Status indications avoid alarmist chaos; instead, they project calibrated forensic certainty through surgical emerald greens, amber warnings, and refined crimson alerts.

## Colors

The palette relies on absolute optical discipline across three structural canvas layers, punctuated by luminous signals that communicate acoustic veracity.

### Foundation & Canvas
- **Void Background (`#090B0E`)**: The baseline infinite canvas.
- **Surface Elevation 1 (`#0E1117`)**: Base module containers and view wrappers.
- **Surface Elevation 2 (`#151921`)**: Raised interaction zones, inspector panels, and cards.
- **Translucent Glass Surface (`rgba(20, 25, 35, 0.72)`)**: Floating navigational planes and overlay modules requiring optical diffusion (`backdrop-blur-xl`).

### Chromatic Signals & Risk States
- **Genuine / Authentic (Primary - `#10B981`)**: Represents biometric match, clean audio provenance, and zero-threat status. Supported by an active deep emerald (`#059669`) and an optical corona (`rgba(16, 185, 129, 0.20)`).
- **Acoustic Waveform & Spectral Telemetry (Secondary - `#06B6D4` & `#6366F1`)**: Reserved exclusively for acoustic waveforms, haptic feedback lines, and machine learning node computations.
- **Moderate Variance / Indeterminate (`#F59E0B`)**: Signals compressed audio, synthetic artifacts below threshold, or missing metadata.
- **Synthetic / Deepfake Alert (Tertiary - `#EF4444`)**: Emits high-risk divergence detection, voice-cloning artifacts, and forensic mismatch markers, complemented by a soft warning halo (`rgba(239, 68, 68, 0.20)`).

### Typography Contrast
- **Primary Text (`#F8FAFC`)**: Ultra-pure high contrast for titles, forensic percentages, and critical findings.
- **Secondary Text (`#94A3B8`)**: Cool slate for structural subtitles, metadata tags, and active axis keys.
- **Muted Text (`#64748B`)**: Subordinate telemetry timestamps, inactive tabs, and structural borders.

## Typography

The typographic hierarchy is structured around functional clarity and forensic rigor. 

- **Primary Interface (Inter)**: Handles all structural text, headlines, and observational readouts. Tight negative letter-spacing on display scales produces a solid, engineered appearance reminiscent of luxury hardware displays.
- **Telemetry & Machine Readouts (JetBrains Mono)**: Employed specifically for confidence intervals, Hz frequencies, file hashes, timestamps, and synthetic probability indexes to enforce an authentic scientific baseline.
- Body copy maintains generous line-height relative to font size to avoid visual crowding in dense diagnostic environments.

## Layout & Spacing

The layout employs an 8pt architectural rhythm, utilizing dynamic margin constraints centered around an edge-to-edge mobile-first paradigm.

### Form Factors & Adaptation
- **Mobile Handset (360px – 640px)**: Single column with `16px` (`space-md`) horizontal safe margins. Content scrolls continuously beneath fixed glass navigational planes. Crucial actions sit in the lower thumb zone directly above the bottom navigation bar.
- **Tablet / Expanded Viewport (641px – 1024px)**: 8-column layout with `24px` (`space-lg`) margins. Telemetry displays split into dual cards (Waveform Analysis alongside Probability Score Card).
- **Desktop Dashboard (1025px+)**: 12-column layout max-width constrained to `1280px`. Left rail anchors navigation; central stage features detailed spectrogram forensics and recording triggers; right rail presents structural metadata and file history.

## Elevation & Depth

Depth is established not through heavy dropshadows, but via atmospheric light refraction and translucent layering.

1. **Sub-surface (Base 0)**: Unadorned `#090B0E`. Houses background ambient canvas glows emitted by live analysis states.
2. **Structural Glass (Layer 1)**: `#141923` rendered with `80%` opacity, paired with a `backdrop-filter: blur(24px)` and a precise `1px` perimeter stroke (`rgba(255, 255, 255, 0.08)`). Top edge strokes carry an amplified highlight (`rgba(255, 255, 255, 0.14)`) to simulate a sharp bevel catching light.
3. **Floating Controls (Layer 2)**: Elevated action modules, contextual menus, and bottom bars utilize `rgba(21, 25, 33, 0.88)` with `backdrop-filter: blur(32px)` and a subtle directional drop shadow: `0 12px 32px -4px rgba(0, 0, 0, 0.65)`.
4. **Active Radiance (Luminous Layer)**: When a recording or authentic sample is engaged, a diffuse, non-directional radial glow radiates behind the element (`box-shadow: 0 0 48px -8px rgba(16, 185, 129, 0.22)`).

## Shapes

The geometric identity balances ergonomics with industrial luxury:
- **Cards and Structural Surfaces**: Feature a refined `16px` (`rounded-lg`) curvature that provides containment without appearing excessively bulbous.
- **Interactive Badges and CTA Buttons**: Utilize full pill geometry (`9999px`) to create tactile, touch-friendly affordances against rigid background modules.
- **Data Scopes and Visualizers**: Maintained with strict, razor-sharp interior inner containers (`8px` radius) to convey scientific accuracy within smooth outer hulls.

## Components

### Buttons
- **Primary Forensic Action (Tactile Record / Run)**: Pill-shaped, deep gradient emerald core (`#10B981` to `#059669`) with an inner top rim glow (`1px inset rgba(255, 255, 255, 0.35)`). Hover/active triggers an expanding concentric ripple wave: two concentric rings pulsing outward from `0%` to `100%` scale at low opacity (`rgba(16, 185, 129, 0.15)`).
- **Secondary Actions**: Frosted translucent pill (`rgba(255, 255, 255, 0.05)`) with an outer hairline stroke (`rgba(255, 255, 255, 0.12)`) and crisp white typography. Hover elevates stroke opacity to `0.25`.

### Circular Authenticity Gauges
- Dual-track SVG circular dials.
- Backing track: Inactive deep track (`rgba(255, 255, 255, 0.06)`), 8px stroke.
- Active value track: Seamless gradient ring shifting according to risk score: Genuine (`#10B981`), Questionable (`#F59E0B`), Deepfake Detected (`#EF4444`). Central readouts present large tabular numbers (`JetBrains Mono`) with risk certainty indicators in uppercase tracking.

### Badges & Risk Chips
- Pill format, height `24px`, padding `0 10px`.
- High-risk: Background `rgba(239, 68, 68, 0.12)`, border `1px solid rgba(239, 68, 68, 0.3)`, text `#EF4444`. Accompanied by a 4px solid status dot.
- Verified: Background `rgba(16, 185, 129, 0.12)`, border `1px solid rgba(16, 185, 129, 0.3)`, text `#10B981`.

### File Ingestion Zone (Acoustic Drop Area)
- Glassmorphic container with an interior dashed vector border (`rgba(255, 255, 255, 0.15)`).
- Drag-over state shifts border to continuous cyan-violet gradient (`#06B6D4` to `#6366F1`) with an ambient internal backlight glow.

### Waveform Visualizer Module
- Enclosed panel framing an dynamic canvas.
- Renders dual-channel high-density frequency bars utilizing `#06B6D4` for baseline audio, with synthetic anomalies mapped in `#EF4444` markers directly over compromised voice segments.

### Mobile Bottom Navigation
- Fixed floating glass capsule docked `16px` above the device safe-area bottom.
- Frosted matrix (`#141923` with 80% opacity, 24px blur) holding four items: `Home`, `Analyze`, `History`, `My Voice`.
- Active tab displays an emerald-illuminated micro dot beneath a clean, line-weight icon, while inactive tabs reside in calm cool slate (`#64748B`).