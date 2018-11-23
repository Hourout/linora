from io import open
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='linora',
      version='0.1.0',
      install_requires=['pandas>=0.20.3', 'scikit-learn>=0.19.1', 'xgboost>=0.81'],
      description='Easy automatic hyperparameter optimization algorithms and libraries for XGBoost and LightGBM.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/linora',
      author='JinQing Lee, Gaojie Wei',
      author_email='hourout@163.com',
      keywords=['hyperparameter-optimization', 'XGBoost', 'LightGBM'],
      license='MIT',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      packages=find_packages(),
      zip_safe=False)
