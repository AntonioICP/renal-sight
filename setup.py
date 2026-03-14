from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='renal_sight',
      version="0.0.1",
      description="AI-assisted kidney stone detection and clinical report generation from CT imaging",
      license="MIT",
      author="Antonio Cisneros",
      author_email="acip0903@gmail.com",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      include_package_data=True,
      zip_safe=False
    )
