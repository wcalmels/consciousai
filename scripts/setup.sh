#!/usr/bin/env bash
# ConsciousAI v3.0 — Setup Script
set -e

echo "=========================================="
echo "  ConsciousAI v3.0 — Setup"
echo "=========================================="

# Check Python
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python $PYTHON_VERSION"

# Install core
echo ""
echo "Installing dependencies..."
pip install numpy --quiet
echo "  ✅ numpy"

# Install Numba (optional but highly recommended)
if python3 -c "import numba" 2>/dev/null; then
    echo "  ✅ numba (already installed)"
else
    echo "  Installing numba (for 10-50× speedup)..."
    pip install numba --quiet && echo "  ✅ numba" || echo "  ⚠️  numba install failed — fallback mode"
fi

# Install scipy
pip install scipy --quiet && echo "  ✅ scipy" || echo "  ⚠️  scipy optional"

# Run quick verification
echo ""
echo "Verifying installation..."
python3 - << 'EOF'
import sys, os
sys.path.insert(0, '.')
try:
    from src.core.engine import IntegratedConsciousnessEngine
    import numpy as np
    cfg_cls = __import__('src.core.engine', fromlist=['IntegratedConfig']).IntegratedConfig
    cfg = cfg_cls()
    cfg.enable_monitoring = False
    cfg.enable_security   = False
    engine = IntegratedConsciousnessEngine(cfg)
    phi = engine.calculate_phi(np.random.rand(50, 5))
    engine.shutdown()
    print(f"  ✅ Engine OK — test Φ = {phi:.4f}")
except Exception as e:
    print(f"  ❌ Engine check failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "  ✅ ConsciousAI v3.0 ready!"
echo ""
echo "  Quick start:"
echo "    python examples/quick_start.py"
echo "    python examples/drone_fleet.py"
echo ""
echo "  Run tests:"
echo "    python -m pytest tests/ -v"
echo "=========================================="
