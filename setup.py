from io import open
from setuptools import setup, find_packages

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

setup(name='linora',
      version='1.0.0rc0',
      install_requires=[
          'pandas>=1.1.0', 
          'pyecharts>=1.6.0',
          'pyecharts_snapshot>=0.1.10', 
          'colorlog>=4.6.2',
          'pillow>=8.0.1'
      ],
      description='Simple and efficient tools for data mining and data analysis.',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://github.com/Hourout/linora',
      author='JinQing Lee',
      author_email='hourout@163.com',
      keywords=['hyperparameter-optimization', 'XGBoost', 'LightGBM', 'data-mining', 'data-analysis', 'machine-learning', 'image', 'text', 'data-science'],
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
          'Programming Language :: Python :: 3.8'
      ],
      packages=find_packages(),
      zip_safe=False)
