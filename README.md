# build-causal-conv1d

Pre-built Linux wheels for
[Causal Conv1d](https://github.com/Dao-AILab/causal-conv1d), across Python,
PyTorch, CUDA, and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index
for each CUDA version. Each wheel has a local version suffix that identifies the
CUDA and PyTorch versions it was built against, such as
`causal-conv1d==1.6.2.post1+cu.12.8.torch.2.11`, and requires the matching
PyTorch release.

Pre-built wheels are available on
[Astral's GPU indexes](https://wheels.astralshosted.com/index.html).
For example, to install a CUDA 12.8 build:

```console
$ uv add causal-conv1d --index astral-cu128=https://wheels.astralshosted.com/simple/cu128/
```

This configures the index and uses it as the source for `causal-conv1d`:

```toml
[tool.uv.sources]
causal-conv1d = { index = "astral-cu128" }

[[tool.uv.index]]
name = "astral-cu128"
url = "https://wheels.astralshosted.com/simple/cu128/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astralshosted.com/simple/cu128/ causal-conv1d
```

## Supported versions

Wheels are available for the following `causal-conv1d` versions:

- [`1.6.2.post1`](https://github.com/astral-sh-build/build-causal-conv1d/releases/tag/v1.6.2.post1)
- [`1.6.1`](https://github.com/astral-sh-build/build-causal-conv1d/releases/tag/v1.6.1.post4)
- [`1.6.0`](https://github.com/astral-sh-build/build-causal-conv1d/releases/tag/v1.6.0-r1)
- [`1.5.4`](https://github.com/astral-sh-build/build-causal-conv1d/releases/tag/v1.5.4-r2)

The latest release, Causal Conv1d 1.6.2.post1, supports the following
combinations:

| PyTorch | Python    | `x86_64` CUDA          | `aarch64` CUDA         |
| ------- | --------- | ---------------------- | ---------------------- |
| 2.4.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.5.1   | 3.9–3.12  | 12.1, 12.4             | —                      |
| 2.6.0   | 3.9–3.12  | 12.4, 12.6             | 12.6                   |
| 2.7.1   | 3.9–3.13  | 12.6, 12.8             | 12.8                   |
| 2.8.0   | 3.9–3.13  | 12.6, 12.8, 12.9       | 12.9                   |
| 2.9.0   | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.10.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |
| 2.11.0  | 3.10–3.14 | 12.6, 12.8, 12.9, 13.0 | 12.6, 12.8, 12.9, 13.0 |

## License

build-causal-conv1d is licensed under the
[Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
