from setuptools import find_packages, setup
from tsmule import __version__ as info

with open('README.md', 'r') as fh:
    long_description = fh.read()

# all folder with __init__.py as packages
package_dir = 'tsmule'
packages = find_packages(
    where=package_dir,
    exclude=('tests*',)
)


setup(name=info.__title__,
      version=info.__version__,
      author=info.__author__,
      author_email=info.__author_email__,
      description=info.__description__,
      long_description=long_description,
      url=info.__url__,
      package_dir={'': package_dir},
      packages=packages,
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6'

      )
