# Contributing to Real-time fMRI AI Analysis

Welcome! 🎉 We're excited you want to contribute to this project. This guide helps different types of contributors understand how they can help improve real-time fMRI analysis for the neuroscience community.

## 🎯 Project Vision

**Core Mission**: AI-powered real-time analysis of fMRI brain data  
**Key Principle**: GPU acceleration is completely optional - the core analysis works everywhere

## 👥 Types of Contributors Welcome

### 🧠 **Neuroscience Researchers**
- **What you bring**: Domain expertise, validation datasets, real-world use cases
- **What you can contribute**: Algorithm validation, scientific accuracy, documentation
- **No need to know**: GPU programming, OpenCL, hardware optimization

### ⚡ **GPU Computing Enthusiasts** 
- **What you bring**: Performance optimization skills, hardware expertise
- **What you can contribute**: OpenCL kernels, multi-GPU algorithms, benchmarks
- **No need to know**: Brain anatomy, fMRI analysis details

### 🤖 **AI/ML Researchers**
- **What you bring**: Deep learning expertise, model architectures
- **What you can contribute**: Better ROI detection, network classification models
- **No need to know**: Hardware specifics, GPU programming

### 🔧 **Software Developers**
- **What you bring**: Clean code, testing, infrastructure
- **What you can contribute**: Architecture improvements, testing, documentation

## 🏗️ Project Architecture

Our **modular architecture** allows you to contribute to specific areas:

```
fmri_core/          ← Neuroscientists focus here
ai_models/          ← AI researchers focus here  
compute_backends/   ← All contributors can help
gpu_acceleration/   ← GPU enthusiasts focus here
visualization/      ← UI/visualization developers
tests/              ← Everyone can contribute tests
```

## 🚀 Quick Start Guide

### For All Contributors

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/realtime-fmri-ai.git
   cd realtime-fmri-ai
   ```

2. **Set up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or: venv\Scripts\activate  # Windows
   
   pip install -e ".[dev]"  # Install in development mode
   ```

3. **Run Tests**
   ```bash
   python tests/run_all_tests.py
   ```

### For Neuroscience Contributors

4. **Focus Areas**
   ```bash
   # Work on core brain analysis
   cd python/
   ls neural_network_analyzer.py brain_atlas_manager.py fmri_data_loader.py
   
   # Add validation datasets
   cd data/
   # Add your fMRI datasets or synthetic data generators
   ```

5. **Test Your Changes**
   ```bash
   python tests/test_core_fmri_analysis.py
   ```

### For GPU Contributors

4. **Focus Areas**
   ```bash
   # Work on GPU acceleration
   cd src/
   ls *.cpp *.cl  # C++ and OpenCL files
   
   # Build GPU components
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j4
   ```

5. **Test GPU Changes**
   ```bash
   python tests/test_gpu_acceleration.py
   ```

### For AI Contributors

4. **Focus Areas**
   ```bash
   # Work on AI models
   cd python/
   ls ai_roi_detector.py realtime_ai_roi_system.py
   
   # Train models (when you have data)
   python examples/train_roi_models.py
   ```

5. **Test AI Changes**
   ```bash
   python tests/test_ai_models.py
   ```

## 📝 Contribution Guidelines

### Code Standards

- **Python**: Follow PEP 8, use type hints
- **C++**: Follow Google C++ Style Guide  
- **Documentation**: All public functions need docstrings
- **Testing**: Add tests for new functionality

### Git Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Keep commits focused and atomic
   - Write clear commit messages
   - Update tests and documentation

3. **Test Everything**
   ```bash
   python tests/run_all_tests.py
   black python/ tests/  # Format code
   flake8 python/ tests/  # Lint code
   ```

4. **Submit Pull Request**
   - Describe what you changed and why
   - Link to relevant issues
   - Include test results

### Pull Request Categories

#### 🧠 **Neuroscience PRs**
- **Focus**: Algorithm accuracy, validation, scientific correctness
- **Required**: Test with real or realistic synthetic data
- **Examples**: "Add partial correlation method", "Validate against published benchmarks"

#### ⚡ **Performance PRs**
- **Focus**: Speed improvements, GPU optimization, memory efficiency  
- **Required**: Benchmark results showing improvement
- **Examples**: "Optimize correlation matrix kernel", "Add multi-GPU load balancing"

#### 🤖 **AI/ML PRs**
- **Focus**: Better models, accuracy improvements, new AI techniques
- **Required**: Model evaluation metrics
- **Examples**: "Improve ROI detection accuracy", "Add attention mechanism"

#### 🔧 **Infrastructure PRs**
- **Focus**: Testing, CI/CD, architecture, documentation
- **Required**: Don't break existing functionality
- **Examples**: "Add performance regression tests", "Improve error handling"

## 🧪 Testing Requirements

### Required Tests
- **Core Functionality**: Must not break basic fMRI analysis
- **Backward Compatibility**: New features should be optional
- **Cross-Platform**: Test on Windows/Linux/macOS when possible

### Test Categories
```bash
# Always run - core functionality
python tests/test_core_fmri_analysis.py

# Optional - run if you changed GPU code
python tests/test_gpu_acceleration.py

# Optional - run if you changed AI code  
python tests/test_ai_models.py

# Comprehensive - run before submitting PR
python tests/run_all_tests.py
```

### Performance Testing
If your change affects performance:
```bash
python examples/performance_benchmark.py
# Include results in your PR description
```

## 📚 Documentation Standards

### Code Documentation
```python
def analyze_brain_connectivity(time_series: np.ndarray, 
                              method: str = 'pearson') -> np.ndarray:
    """
    Compute brain connectivity matrix from fMRI time series.
    
    Args:
        time_series: fMRI data [timepoints x regions]
        method: Correlation method ('pearson', 'partial')
        
    Returns:
        Connectivity matrix [regions x regions]
        
    Example:
        >>> data = np.random.randn(200, 100)  # 200 timepoints, 100 regions
        >>> connectivity = analyze_brain_connectivity(data)
        >>> print(connectivity.shape)  # (100, 100)
    """
```

### README Updates
- Update installation instructions if you add dependencies
- Add usage examples for new features
- Keep the "Quick Start" section working

## 🎯 Specific Contribution Areas

### 🧠 For Neuroscientists

#### High-Impact Contributions:
1. **Algorithm Validation**
   - Compare against established tools (FSL, SPM, AFNI)
   - Test with real datasets
   - Validate network metrics

2. **Brain Atlas Integration**
   - Add support for new atlases
   - Improve coordinate mapping
   - Better region labeling

3. **Quality Control**
   - Motion correction algorithms
   - Artifact detection
   - Data validation

#### Getting Started:
```python
# Example: Add new correlation method
def compute_partial_correlation(time_series):
    """Your new correlation method here"""
    # Implement partial correlation
    # Add to neural_network_analyzer.py
```

### ⚡ For GPU Developers  

#### High-Impact Contributions:
1. **Multi-GPU Optimization**
   - Better load balancing
   - Memory management
   - Cross-GPU communication

2. **New Hardware Support**
   - NVIDIA GPU support
   - Intel GPU support  
   - Specialized AI accelerators

3. **Performance Kernels**
   - Optimized OpenCL kernels
   - Memory access patterns
   - Async execution

#### Getting Started:
```cpp
// Example: Optimize correlation kernel
__kernel void compute_correlation_optimized(
    __global const float* time_series,
    __global float* correlation_matrix,
    const int n_regions,
    const int n_timepoints
) {
    // Your optimized kernel here
    // Add to kernels/ directory
}
```

### 🤖 For AI Researchers

#### High-Impact Contributions:
1. **Better ROI Detection**
   - Higher accuracy models
   - Real-time inference optimization
   - Multi-modal integration

2. **Network Classification**  
   - Better brain state detection
   - Temporal modeling
   - Attention mechanisms

3. **Model Training Pipeline**
   - Training scripts
   - Data augmentation
   - Transfer learning

#### Getting Started:
```python
# Example: Improve ROI model architecture
class ImprovedROIDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Your improved architecture
        # Add to ai_roi_detector.py
```

### 🔧 For Software Developers

#### High-Impact Contributions:
1. **Testing Infrastructure**
   - Automated testing
   - Performance regression tests
   - Cross-platform CI/CD

2. **API Design**
   - Cleaner interfaces
   - Better error handling
   - Improved documentation

3. **Deployment Tools**
   - Docker containers  
   - Cloud deployment
   - Package distribution

## 🐛 Bug Reports

### Good Bug Reports Include:
1. **Environment**: OS, Python version, GPU info
2. **Reproduction Steps**: Exact commands that cause the issue
3. **Expected vs Actual**: What should happen vs what happens
4. **Logs**: Full error messages and stack traces

### Bug Report Template:
```markdown
**Environment:**
- OS: Windows 11 / Ubuntu 22.04 / macOS 13
- Python: 3.11.0
- GPU: AMD Radeon RX 7700S
- Package Version: 0.1.0

**Steps to Reproduce:**
1. Install with `pip install realtime-fmri-ai[gpu]`
2. Run `python -c "from realtime_fmri_ai import fMRIAnalyzer; analyzer = fMRIAnalyzer()"`
3. Error occurs

**Expected Behavior:**
Should initialize without error

**Actual Behavior:**
```
ImportError: No module named 'pyopencl'
```

**Additional Context:**
Happens only with GPU installation, core installation works fine.
```

## 💡 Feature Requests

### Good Feature Requests:
- **Scientific Justification**: Why is this needed?
- **Use Cases**: How would researchers use this?
- **Implementation Ideas**: Any thoughts on how to implement?

### Feature Request Template:
```markdown
**Scientific Background:**
fMRI researchers need to analyze dynamic connectivity changes in real-time...

**Proposed Feature:**
Add sliding window correlation analysis with configurable window sizes...

**Use Case:**
During neurofeedback sessions, researchers need to detect when...

**Implementation Notes:**
Could be implemented as an extension to the existing correlation module...
```

## 🚦 Code Review Process

### What We Look For:
1. **Correctness**: Does the code work as intended?
2. **Testing**: Are there appropriate tests?
3. **Documentation**: Is the code well-documented?
4. **Performance**: Does it maintain or improve performance?
5. **Architecture**: Does it fit the modular design?

### Review Timeline:
- **Initial Response**: Within 2 business days
- **Full Review**: Within 1 week for small changes, 2 weeks for large features
- **Merge**: After approval and passing all tests

## 🎉 Recognition

### Contributors Get:
- **Credit**: Listed in CONTRIBUTORS.md and release notes
- **Badge**: Contributor badge on GitHub profile
- **Swag**: Project stickers for significant contributions (when available)

### Maintainer Path:
Consistent, high-quality contributors may be invited to become maintainers with:
- Commit access to main repository
- Input on project direction
- Recognition at conferences

## 📞 Getting Help

### Questions?
- **GitHub Discussions**: For design questions and ideas
- **GitHub Issues**: For bugs and specific feature requests
- **Email**: maintainers@realtime-fmri-ai.org for sensitive topics

### Real-time Chat:
- **Discord**: [Join our community](https://discord.gg/realtime-fmri-ai) (when available)
- **Office Hours**: Monthly video calls for contributors

## 🏆 Top Contribution Opportunities

### Immediate Impact (Easy):
1. 📝 Improve documentation and examples
2. 🧪 Add more test cases
3. 🎨 Create visualization examples
4. 📊 Add performance benchmarks

### High Impact (Medium):
1. 🧠 Validate algorithms against published papers
2. ⚡ Optimize GPU kernels for better performance  
3. 🤖 Improve AI model accuracy
4. 📱 Add real-time visualization dashboard

### Transformative (Hard):
1. 🔬 Integration with real fMRI scanners
2. 🌐 Cloud-based processing pipeline
3. 📡 Real-time streaming protocol
4. 🧬 Multi-modal brain analysis (fMRI + EEG)

---

## 🙏 Thank You!

Your contributions help advance real-time brain analysis for researchers worldwide. Whether you fix a typo, optimize a kernel, or validate an algorithm - every contribution matters.

**Together, we're making neuroscience research faster, more accessible, and more powerful! 🧠⚡🚀** 