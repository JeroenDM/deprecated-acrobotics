from setuptools import setup, find_packages, Extension
import numpy

graph_module = Extension('_graph',
    language="c++",
    extra_compile_args=['-std=c++11'],
    sources=['acrobotics/cpp/graph.i', 'acrobotics/cpp/src/graph.cpp'],
    include_dirs=['acrobotics/cpp/include', '/usr/include/eigen3', numpy.get_include()],
    swig_opts=['-c++', '-I acrobotics/cpp']
    )

setup(
    name = 'acrobotics',
    version = '0.1',
    packages=find_packages(),
    include_package_data=True,
    description = 'Acro Robotics',
    long_description=('Tools to write motion planning algorithms' +
    'for robot arms.'),
    author = 'Jeroen De Maeyer',
    author_email = 'jeroen.demaeyer@kuleuven.be',
    url = 'https://github.com/JeroenDM/acrobotics',
    download_url = 'todo',
    keywords = ['robotics', 'motion planning'],
    classifiers = [],
    install_requires=['numpy', 'matplotlib', 'pyquaternion'],
    python_requires='>=3',
    ext_package='acrobotics.cpp',
    ext_modules=[graph_module],
)
