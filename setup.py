from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

__version__ = "0.0.1"

# 检查CUDA是否可用
enable_cuda = torch.cuda.is_available()

# 获取源文件列表
def get_extensions():
    ext_modules = []
    
    if enable_cuda:
        # CUDA扩展
        ext_modules.append(
            CUDAExtension(
                name="torch_ops._C",
                sources=[
                    "src/bindings.cpp",
                    "src/square_plus.cpp",
                    "src/modulo.cpp",
                    "src/square_plus_cuda.cu",
                    "src/modulo_cuda.cu",
                ],
                include_dirs=[
                    "src",
                ],
                define_macros=[("WITH_CUDA", None), ("VERSION_INFO", __version__)],
            )
        )
    else:
        # 仅CPU扩展
        ext_modules.append(
            CppExtension(
                name="torch_ops._C",
                sources=[
                    "src/bindings.cpp",
                    "src/square_plus.cpp",
                    "src/modulo.cpp",
                ],
                include_dirs=[
                    "src",
                ],
                define_macros=[("VERSION_INFO", __version__)],
            )
        )
    
    return ext_modules

setup(
    name="torch_ops",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourname/torch_ops",
    description="PyTorch extension with custom CUDA operators",
    long_description="",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    python_requires=">=3.6",
    packages=["torch_ops"],
)