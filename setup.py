from io import open
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

requires_list=['pandas>=0.24.1', 'scikit-learn>=0.20.2', 'xgboost>=0.81', 'pyecharts>=0.5.11',
              'pyecharts_snapshot>=0.1.10', 'numpy>=1.16.2']

setup(name='linora',
      version='0.4.0',
      install_requires=requires_list,
      description='Easy automatic hyperparameter optimization algorithms and libraries for XGBoost and LightGBM.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/linora',
      author='JinQing Lee, Gaojie Wei',
      author_email='hourout@163.com',
      keywords=['hyperparameter-optimization', 'XGBoost', 'LightGBM'],
      license='Apache License Version 2.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7'
      ],
      packages=find_packages(),
      zip_safe=False)
