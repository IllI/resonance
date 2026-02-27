#!/usr/bin/env python3
"""
Script to add fallback ROIDetection class to brain_3d_model_generator.py
"""

import re

# Read the file
with open('brain_3d_model_generator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the pattern to replace
old_pattern = '''except ImportError:
    HAS_LOCAL_MODULES = False
    warnings.warn("Local modules not available - limited functionality")

@dataclass
class AnatomicalFeature:'''

new_pattern = '''except ImportError:
    HAS_LOCAL_MODULES = False
    warnings.warn("Local modules not available - limited functionality")
    
    # Fallback ROIDetection dataclass when ai_roi_detector is not available
    @dataclass
    class ROIDetection:
        """Fallback ROIDetection when ai_roi_detector is not available."""
        region_id: int = 0
        region_name: str = ""
        confidence: float = 0.0
        coordinates: Tuple[int, int, int] = (0, 0, 0)
        network_state: str = "unknown"
        activation_strength: float = 0.0
        temporal_pattern: np.ndarray = None

@dataclass
class AnatomicalFeature:'''

# Replace the pattern
if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    
    # Write the modified content back
    with open('brain_3d_model_generator.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Successfully added fallback ROIDetection class!")
else:
    print("Pattern not found in file - may already be modified or file structure is different")
    print("Looking for alternate patterns...")
    
    # Try to find just the except block
    if 'except ImportError:\n    HAS_LOCAL_MODULES = False' in content:
        print("Found the except block")
    else:
        print("Could not find the except block pattern")
