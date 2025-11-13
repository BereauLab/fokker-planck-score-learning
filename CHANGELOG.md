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
- Add support for py3.14

### Other changes:
- Use `cosine_decay_schedule` instead of `warmup_cosine_decay_schedule` if user selects `warmup_steps=0`

### Bugfix 🐛:
- Exclude docs directory from build


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
