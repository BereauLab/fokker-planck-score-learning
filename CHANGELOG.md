# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and [Element](https://github.com/vector-im/element-android)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[//]: # (Available sections in changelog)
[//]: # (### API changes warning ⚠️:)
[//]: # (### Added Features and Improvements 🙌:)
[//]: # (### Bugfix 🐛:)
[//]: # (### Other changes:)


## [Unreleased]
### Added Features and Improvements 🙌:
- Added submodule to work with MD data `fpsl.utils.md` together new with tutorial 🎉
- Added submodule with periodic fourier expansion `fpsl.utils.fourier`
- Generalized the Fokker-Planck energy regularization to arbitrary dimensions via a
  stochastic finite-difference estimator, enabled through the new `v_sigma` parameter
  of `FPSL` (falls back to the previous 1D exact-gradient estimator when `v_sigma=0`)
- Added `fpsl.utils.fourier.get_fourier_func2d`, a 2D analog of `get_fourier_func`
- Generalized the end-of-training Weights & Biases free-energy-landscape logging to
  1D and 2D data (previously hardcoded to 1D)
- Added a tutorial section on enforcing custom symmetries in the score network

### Other changes:
- Use `cosine_decay_schedule` instead of `warmup_cosine_decay_schedule` if user selects `warmup_steps=0`

### Bugfix 🐛:
- Exclude docs directory from build
- Fixed `fpsl.utils.md.load_trajs` discarding the loaded pull-force trajectory
- Fixed `ScorePeriodicMLP` silently dropping its sine Fourier features (falling back
  to the same cosine-only behavior as `ScoreSymmetricPeriodicMLP`)


## [0.2.0] - 2025-08-06
### Added Features and Improvements 🙌:
- Added CI for automated publishing to PyPi
- Added a tutorial with a brief theoretical recap, and an example application
- Improved documentation, with submodule docstrings

### Other changes:
- Cleaned up some code
- Add dependabot

### Bugfix 🐛:
- Fix using inner product when estimating work


## [0.1.0] - 2025-06-28
- Initial release 🎉


[Unreleased]: https://github.com/bereaulab/fokker-planck-score-learning/compare/v0.2.0...main
[0.2.0]: https://github.com/bereaulab/fokker-planck-score-learning/compare/v0.1.0...0.2.0
[0.1.0]: https://github.com/bereaulab/fokker-planck-score-learning/tree/v0.1.0
