from setuptools import find_packages, setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

# all folder with __init__.py as packages
packages = find_packages(
    exclude=('tests*',)
)

setup(name='tsmule',
      version='0.0.1',
      author='tsmule',
      author_email='tsmule',
      description='tsmule',
      long_description=long_description,
      url='https://github.com/dbvis-ukon/ts-mule',
    #   package_dir={'': 'src'},
      packages=packages,
      classifiers=[
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
      ],
      python_requires='>=3.6'

      )
