import io
from setuptools import setup, find_packages


def readme():
    with io.open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='linora',
      version='2.0.0rc2',
      install_requires=[
          'pandas>=1.3.5', 
          'Pillow>=9.5.0',
          'joblib>=1.3.2',
          'requests>=2.28.0',
          'rarfile',
          'av'
      ],
      description='Simple and efficient tools for data mining and data analysis.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/linora',
      author='JinQing Lee',
      author_email='hourout@163.com',
      keywords=['hyperparameter-optimization', 'XGBoost', 'LightGBM', 'data-mining', 
                'data-analysis', 'machine-learning', 'image', 'text', 'data-science', 
                'logging', 'parallel', 'feature-engineering', 'metrics', 'schedulers', 'datasets'],
      license='Apache License Version 2.0',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Visualization',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11'
      ],
      packages=find_packages(),
      zip_safe=False)
