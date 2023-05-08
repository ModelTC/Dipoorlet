import os
import re
import setuptools
from dipoorlet import __version__


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


def get_package_suffix():
    package_suffix = __version__
    if os.environ.get('CI_COMMIT_REF_SLUG', None):
        ci_commit_ref_slug = os.environ['CI_COMMIT_REF_SLUG']
        if not ci_commit_ref_slug.startswith(
                'release') and not ci_commit_ref_slug.startswith('master'):
            ci_commit_ref_slug = re.sub(r'[^A-Za-z0-9._-]', '_',
                                        ci_commit_ref_slug)
            package_suffix += f".{ci_commit_ref_slug}"
            if ci_commit_ref_slug == "dev" and os.environ.get('CI_COMMIT_SHORT_SHA', None):
                ci_commit_short_sha = os.environ['CI_COMMIT_SHORT_SHA']
                package_suffix += f".{ci_commit_short_sha}"
    return package_suffix


setuptools.setup(
    name="dipoorlet",
    version=get_package_suffix(),
    author="RD-MTC",
    description=("Offline quantization and profiling."),
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    install_requires=read_requirements()
)
