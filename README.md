# Julia package for working with scientific detectors

`ScientificDetectors` is a  [Julia][julia-url] package for calibrating and pre-processing
data from scientific detectors.


## Installation

`ScientificDetectors` is not yet an [official Julia package][julia-pkgs-url] so you
have to clone the repository.  In Julia, hit the `]` key to switch to the
package manager REPL (you should get a `... pkg>` prompt) and type:

```julia
pkg> add https://github.com/emmt/ScientificDetectors.jl#master
```

if you use HTTPS, or:

```julia
pkg> add git@github.com:emmt/ScientificDetectors.jl#master
```

if you use SSH.  If you do not want to follow the master version, remove the
`#master` prefix in the URLs above.

The above commands may fail because some other *non-official* packages
([ArrayTools](https://github.com/emmt/ArrayTools.jl) and
[EasyFITS](https://github.com/emmt/EasyFITS.jl)) are required.  The recommanded
way to install them (assuming SSH) is to type:

```julia
pkg> add https://github.com/emmt/ArrayTools.jl#master
pkg> add https://github.com/emmt/EasyFITS.jl#master
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/ScientificDetectors.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/ScientificDetectors.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/ScientificDetectors.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/ScientificDetectors.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/ScientificDetectors.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/ScientificDetectors-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/ScientificDetectors.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/ScientificDetectors.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/ScientificDetectors.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/ScientificDetectors.jl?branch=master

[julia-url]: https://julialang.org/
[julia-pkgs-url]: https://pkg.julialang.org/
