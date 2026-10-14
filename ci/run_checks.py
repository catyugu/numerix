#!/usr/bin/env python3
"""Build, test, install, relocate and embed numerix on every CI platform."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=["Debug", "Release"], default="Release")
    parser.add_argument("--generator", default=os.environ.get("CMAKE_GENERATOR"))
    parser.add_argument("--build-root", type=Path, default=ROOT / "build-ci")
    args = parser.parse_args()
    work = args.build_root.resolve()
    cache = work / "cpm-cache"
    original = work / "install-original"
    prefix = work / "install-relocated"

    def configure(source, build, *options):
        generator = ["-G", args.generator] if args.generator else []
        run("cmake", "-S", source, "-B", build, *generator,
            f"-DCMAKE_BUILD_TYPE={args.config}", f"-DCPM_SOURCE_CACHE={cache}", *options)

    def check(build):
        run("cmake", "--build", build, "--config", args.config, "--parallel", "2")
        run("ctest", "--test-dir", build, "-C", args.config,
            "--output-on-failure", "--no-tests=error")

    library = work / "library"
    configure(ROOT, library, f"-DCMAKE_INSTALL_PREFIX={original}",
              "-DNUMERIX_WARNINGS_AS_ERRORS=ON")
    check(library)
    run("cmake", "--install", library, "--config", args.config)
    if prefix.exists():
        shutil.rmtree(prefix)
    shutil.move(str(original), str(prefix))
    configure(ROOT / "ci/consumer", work / "consumer", f"-DCMAKE_PREFIX_PATH={prefix}")
    check(work / "consumer")

    # Reuse the installed dependencies: also exercises NUMERIX_USE_SYSTEM_KOKKOS.
    configure(ROOT / "ci/superproject", work / "superproject",
              f"-DCMAKE_PREFIX_PATH={prefix}", "-DNUMERIX_USE_SYSTEM_KOKKOS=ON",
              "-Werror=dev")
    check(work / "superproject")


if __name__ == "__main__":
    main()
