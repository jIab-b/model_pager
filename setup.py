from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

setup(
    name='page_table_ext',
    version='0.1',
    ext_modules=[
        CUDAExtension(
            name='page_table_ext',
            sources=[
                'csrc/weight_pager.cu',
                'csrc/um_tensor.cpp',
                'csrc/page_table.cpp',
            ],
            include_dirs=['csrc'],
            extra_link_args=[f"-Wl,-rpath,{os.path.join(os.path.dirname(torch.__file__), 'lib')}"],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
