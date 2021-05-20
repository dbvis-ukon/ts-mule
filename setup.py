from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

# all folder with __init__.py as packages
package_dir = 'tsmule'
packages = find_packages(
    where=package_dir,
    exclude=('tests*',)
)

package_name = "tsmule"
author = "Udo Schlegel, Duy Lam Vo"
author_email = "u.schlegel@uni-konstanz.de, duy-lam.vo1102@outlook.com"
description = package_name
version = '0.0.1'
url = 'https://github.com/dbvis-ukon/ts-mule'

setup(name=package_name,
      version=version,
      author=author,
      author_email=author_email,
      description=description,
      long_description=long_description,
      url=url,
      package_dir={'': package_dir},
      packages=packages,
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6'

      )
