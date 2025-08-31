#!/usr/bin/env python3
"""
Setup script for Real-time fMRI AI Analysis

Supports modular installation:
- Core: Basic fMRI analysis (works everywhere)
- GPU: Adds OpenCL acceleration
- AI: Adds PyTorch AI models
- Full: Complete installation with all features
"""

from setuptools import setup, find_packages, Extension
import os
import sys
import subprocess
from pathlib import Path

# Read version from file
def get_version():
    """Get version from version.py file."""
    version_file = Path(__file__).parent / "python" / "version.py"
    if version_file.exists():
        with open(version_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

# Read README for long description
def get_long_description():
    """Get long description from README."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, encoding='utf-8') as f:
            return f.read()
    return "Real-time fMRI AI Analysis with optional GPU acceleration"

# Core dependencies (always installed)
CORE_REQUIREMENTS = [
    "numpy>=1.24.0",
    "scipy>=1.10.0", 
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "nibabel>=5.1.0",
    "nilearn>=0.10.1",
    "networkx>=3.1",
    "plotly>=5.15.0",
    "tqdm>=4.65.0",
    "psutil>=5.9.0"
]

# Optional dependency groups
EXTRAS_REQUIRE = {
    # GPU acceleration support
    "gpu": [
        "pyopencl>=2023.1.4",
        # Note: OpenCL drivers must be installed separately
    ],
    
    # AI models support  
    "ai": [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-learn>=1.3.0",
        "transformers>=4.30.0",  # For advanced AI models
    ],
    
    # Advanced neuroimaging features
    "neuro": [
        "siibra-python>=0.4.12",
        "brainglobe-atlasapi>=2.0.0", 
        "ANTsPy>=0.3.2",
        "templateflow>=23.1.0",
        "mne>=1.4.0"
    ],
    
    # 3D visualization enhancements
    "viz": [
        "mayavi>=4.8.1",
        "vtk>=9.2.0", 
        "pyvista>=0.42.0",
        "ipywidgets>=8.0.0",
        "bokeh>=3.2.0"
    ],
    
    # Real-time processing optimizations
    "realtime": [
        "asyncio",
        "zmq",
        "redis>=4.5.0"  # For real-time data streaming
    ],
    
    # Development tools
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
        "sphinx>=7.1.0",
        "wheel>=0.40.0",
        "build>=0.10.0"
    ]
}

# Convenience combinations
EXTRAS_REQUIRE.update({
    # Multi-GPU setup (our AMD configuration)
    "multi-gpu": EXTRAS_REQUIRE["gpu"] + ["concurrent.futures"],
    
    # Complete AI package
    "ai-full": EXTRAS_REQUIRE["ai"] + EXTRAS_REQUIRE["viz"],
    
    # Research package
    "research": EXTRAS_REQUIRE["neuro"] + EXTRAS_REQUIRE["viz"] + EXTRAS_REQUIRE["ai"],
    
    # Everything
    "full": (CORE_REQUIREMENTS + 
             EXTRAS_REQUIRE["gpu"] + 
             EXTRAS_REQUIRE["ai"] + 
             EXTRAS_REQUIRE["neuro"] + 
             EXTRAS_REQUIRE["viz"] + 
             EXTRAS_REQUIRE["realtime"]),
    
    # Testing
    "test": EXTRAS_REQUIRE["dev"] + ["coverage>=7.0.0"]
})

# C++ Extensions (optional GPU acceleration)
def get_opencl_extensions():
    """Get OpenCL C++ extensions if available."""
    extensions = []
    
    # Check if OpenCL development files are available
    try:
        import pyopencl
        
        # Define OpenCL extension
        opencl_ext = Extension(
            name="realtime_fmri_ai.gpu_acceleration.brain_gpu_lib",
            sources=[
                "src/brain_gpu_accelerator.cpp",
                "src/dual_gpu_brain_accelerator.cpp"
            ],
            include_dirs=[
                "src/",
                "/usr/include/CL",  # Linux
                "/System/Library/Frameworks/OpenCL.framework/Headers",  # macOS
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/include",  # Windows CUDA
            ],
            libraries=["OpenCL"],
            library_dirs=[
                "/usr/lib",
                "/usr/local/lib",
                "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/lib/x64"
            ],
            define_macros=[
                ("CL_TARGET_OPENCL_VERSION", "200"),
                ("BUILDING_DLL", None)
            ],
            language="c++"
        )
        
        extensions.append(opencl_ext)
        
    except ImportError:
        print("⚠️ PyOpenCL not available - GPU extensions will not be built")
    
    except Exception as e:
        print(f"⚠️ OpenCL extension setup failed: {e}")
    
    return extensions

# Custom build command that handles optional extensions
class OptionalBuildExt:
    """Build extensions optionally - don't fail if OpenCL unavailable."""
    
    def __init__(self):
        self.extensions = get_opencl_extensions()
    
    def build_extension(self, ext):
        try:
            # Attempt to build extension
            super().build_extension(ext)
            print(f"✅ Built extension: {ext.name}")
            
        except Exception as e:
            print(f"⚠️ Failed to build extension {ext.name}: {e}")
            print("   GPU acceleration will not be available")

# Setup configuration
setup(
    name="realtime-fmri-ai",
    version=get_version(),
    description="AI-powered real-time fMRI analysis with optional GPU acceleration",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    
    # Author and contact
    author="Real-time fMRI AI Team",
    author_email="contact@realtime-fmri-ai.org",
    url="https://github.com/your-org/realtime-fmri-ai",
    
    # License and classifiers
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
        "Environment :: GPU",
    ],
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Package discovery
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "realtime_fmri_ai": [
            "data/*.json",
            "kernels/*.cl", 
            "models/*.pt",
            "atlases/*.nii.gz"
        ]
    },
    
    # Dependencies
    install_requires=CORE_REQUIREMENTS,
    extras_require=EXTRAS_REQUIRE,
    
    # Optional C++ extensions
    ext_modules=get_opencl_extensions(),
    zip_safe=False,  # Required for C extensions
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "fmri-realtime=realtime_fmri_ai.cli:main",
            "fmri-gpu-info=realtime_fmri_ai.gpu_info:main",
            "fmri-benchmark=realtime_fmri_ai.benchmark:main",
        ]
    },
    
    # Keywords for PyPI search
    keywords=[
        "fmri", "neuroscience", "brain", "real-time", "ai", "machine-learning", 
        "gpu", "opencl", "neuroimaging", "connectivity", "networks"
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://realtime-fmri-ai.readthedocs.io/",
        "Source": "https://github.com/your-org/realtime-fmri-ai",
        "Bug Reports": "https://github.com/your-org/realtime-fmri-ai/issues",
        "Funding": "https://github.com/sponsors/your-org",
    }
)

# Post-install message
print("""
🎉 Real-time fMRI AI Analysis installed successfully!

📦 Installation options:
   Core: pip install realtime-fmri-ai
   GPU:  pip install realtime-fmri-ai[gpu]
   AI:   pip install realtime-fmri-ai[ai] 
   Full: pip install realtime-fmri-ai[full]

🚀 Quick start:
   from realtime_fmri_ai import fMRIAnalyzer
   analyzer = fMRIAnalyzer()  # Auto-selects best backend

📖 Documentation: https://realtime-fmri-ai.readthedocs.io/
🐛 Issues: https://github.com/your-org/realtime-fmri-ai/issues
""") 