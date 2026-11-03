# koyo
Grab-bag of reusable Python helpers (CLI, logging, timing, multiprocessing, config, figure export, spectra) shared across all projects.

## Role in ecosystem
Foundation utility library with no internal deps; almost every other library imports it.
Internal deps: none
Depended on by: annotine, autoims-core, image2image-io, image2image-reg, ims-utils, imspy, imspy-data, imspy-ionglow, imspy-misc, imspy-multi, imspy-pasef, imspy-pre, imzy, pyspa, qtextra, qtextraplot, qucee, yoki5 (imported, undeclared: image2image, imspy-core, imspy-quant, imspy-stats, imspy-train, ionglow)

## Key abstractions
- `koyo.typing` (`PathLike`, ...), `koyo.utilities`, `koyo.path`, `koyo.system` — general helpers
- `koyo.timer` (`MeasureTimer`, `Timer`), `koyo.logging` (`LoggerMixin`, loguru setup)
- `koyo.multicore` — `MultiCoreExecutor`, `MultiCoreDispatcher`, `Parallel`
- `koyo.click` (`Parameter`, `Parameters`, verbosity/env helpers), `koyo.config.BaseConfig`
- `koyo.json_cache.JSONCache`, `koyo.containers` (`SizedDict`, `MutableSequence`, ...)
- `koyo.fig_mixin.FigureMixin` (+ `PDFMixin`, `PPTXMixin`) — export figures to PDF/PPTX
- `koyo.spectrum`, `koyo.sparse`, `koyo.numpy` — array and spectrum helpers

## Do not re-explain
When working in this repo, assume familiarity with: none (no internal deps)
