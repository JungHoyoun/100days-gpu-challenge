from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_flash',
    ext_modules=[
        CUDAExtension(
            name='custom_flash',
            sources=[
                'src/custom_flash.cc',
                'src/custom_flash.cuh',
                'src/custom_flash.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)