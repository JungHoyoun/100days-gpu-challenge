from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash',
    ext_modules=[
        CUDAExtension(
            name='',
            sources=[
                'src/custom_flash.cc',
                'src/custom_flash.cuh',
                'src/custom_flash.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)