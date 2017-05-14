from setuptools import find_packages, setup

setup(
    name='EventDetection',
    version='1.0',
    packages=find_packages(),
    author='Tomas Kala',
    author_email='kalatoma@fel.cvut.cz',
    install_requires=['numpy >= 1.11.3',
                      'scipy >= 0.18.1',
                      'scikit-learn >= 0.18.1',
                      'matplotlib >= 2.0.0',
                      'wordcloud >= 1.2.1',
                      'gensim >= 1.0.1']
)
