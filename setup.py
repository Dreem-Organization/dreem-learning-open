import subprocess
import setuptools

# commit_hash = subprocess.check_output(["git", "rev-parse", "--verify", "--short", "HEAD"]).strip().decode("utf-8")
latest_version = "1.2.1"

setuptools.setup(
    name="dreem-learning-open",
    version=latest_version,
    author="Dreem",
    author_email="antoine@dreem.com",
    description="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
