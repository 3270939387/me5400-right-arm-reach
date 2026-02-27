from setuptools import setup, find_packages


setup(
    name="surgery_needle",
    version="0.1.0",
    description="Surgery needle reach task for Isaac Lab (manager-based, rl_games).",
    author="Your Team",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_assets",
        "isaaclab_rl",
    ],
    python_requires=">=3.10",
    zip_safe=False,
)
