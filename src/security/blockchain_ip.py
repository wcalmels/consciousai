#!/usr/bin/env python3
"""
Blockchain IP Protection Script
================================

Timestamps your intellectual property on blockchain for legal proof of:
- Date of invention
- Authorship
- Prior art

Author: Walter Calmels
Date: 2024-11-26
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of file"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def create_invention_document(config):
    """Create structured invention disclosure document"""
    
    # Calculate hashes of key files
    file_hashes = {}
    for file_path in config['files_to_protect']:
        if Path(file_path).exists():
            file_hashes[file_path] = calculate_file_hash(file_path)
        else:
            print(f"⚠️  Warning: File not found: {file_path}")
    
    # Create invention document
    invention = {
        "metadata": {
            "version": "1.0",
            "created": datetime.utcnow().isoformat() + "Z",
            "creator": config['inventor_name'],
            "purpose": "Legal proof of invention date and authorship"
        },
        "invention": {
            "title": config['invention_title'],
            "abstract": config['invention_abstract'],
            "inventor": config['inventor_name'],
            "date_of_conception": config.get('conception_date', datetime.now().date().isoformat()),
        },
        "technical_details": {
            "problem_solved": config.get('problem', ''),
            "solution_overview": config.get('solution', ''),
            "novelty_claims": config.get('novelty_claims', []),
            "technical_advantages": config.get('advantages', [])
        },
        "files": {
            "description": "SHA-256 hashes of files containing the invention",
            "hashes": file_hashes
        },
        "commercial_value": {
            "applications": config.get('applications', []),
            "target_markets": config.get('markets', [])
        }
    }
    
    return invention

def save_invention_document(invention, output_path):
    """Save invention document as JSON"""
    with open(output_path, 'w') as f:
        json.dump(invention, f, indent=2, sort_keys=True)
    print(f"✅ Invention document saved: {output_path}")
    return output_path

def calculate_document_hash(invention_path):
    """Calculate hash of invention document"""
    return calculate_file_hash(invention_path)

def generate_opentimestamps_commands(files_to_timestamp):
    """Generate OpenTimestamps commands"""
    
    print("\n" + "="*70)
    print("📝 OPENTIMESTAMPS COMMANDS (FREE)")
    print("="*70)
    
    print("\n1. Install OpenTimestamps (if not installed):")
    print("   pip install opentimestamps-client")
    
    print("\n2. Timestamp your files:")
    for file_path in files_to_timestamp:
        print(f"   ots stamp {file_path}")
    
    print("\n3. This will create .ots files (proofs)")
    print("   Keep these .ots files safe!")
    
    print("\n4. Verify at any time:")
    for file_path in files_to_timestamp:
        print(f"   ots verify {file_path}.ots")
    
    print("\n5. Verify online:")
    print("   https://opentimestamps.org/")
    
    return True

def generate_ethereum_example(invention_hash):
    """Generate Ethereum timestamping example (for reference)"""
    
    print("\n" + "="*70)
    print("💎 ETHEREUM TIMESTAMPING (OPTIONAL, ~$5-20)")
    print("="*70)
    
    print(f"""
Note: This is for reference. You'll need:
- Ethereum wallet with ETH
- Web3 provider (Infura, Alchemy)

Example code (DO NOT RUN without reviewing):

from web3 import Web3

# Your invention hash
INVENTION_HASH = "{invention_hash}"

# Connect to Ethereum
w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_KEY'))

# Create transaction with hash in data
tx = {{
    'to': '0x0000000000000000000000000000000000000000',
    'value': 0,
    'gas': 21000 + len(INVENTION_HASH.encode()) * 68,
    'gasPrice': w3.eth.gas_price,
    'data': '0x' + INVENTION_HASH.encode().hex()
}}

# Sign and send (requires private key)
# tx_hash = ...

# Save receipt for legal proof
print(f"Transaction: {{tx_hash}}")
print(f"Timestamp: {{block_timestamp}}")
""")

def main():
    """Main execution"""
    
    print("="*70)
    print("🔐 BLOCKCHAIN IP PROTECTION TOOLKIT")
    print("="*70)
    print("Protects your intellectual property with blockchain timestamping")
    print()
    
    # Configuration
    config = {
        'inventor_name': 'Walter Calmels',
        'invention_title': 'Enhanced Consciousness Framework with JIT Compilation and Self-Repair',
        'invention_abstract': '''
Novel implementation of Integrated Information Theory with:
1. Just-In-Time (JIT) compilation using Numba (10-50× speedup)
2. Self-repairing cache mechanism (50× failure reduction)
3. Domain-specific preprocessing (5-15% accuracy improvement)
4. Auto-vigilance monitoring system

Enables real-time consciousness measurement in autonomous systems
with up to 200 components.
        '''.strip(),
        
        'problem': 'Existing IIT implementations have O(2^N) complexity and fail with corrupted data',
        
        'solution': '''
Spectral decomposition (O(N^2 log N)) combined with:
- Numba JIT for 10-50× speedup in covariance computation
- Self-repairing cache that auto-detects and removes invalid entries
- Domain-specific preprocessing for quantum, biological, neural systems
- Continuous auto-vigilance monitoring
        '''.strip(),
        
        'novelty_claims': [
            'First implementation of IIT with JIT compilation',
            'First self-repairing cache for consciousness metrics',
            'First domain-specific preprocessing for Φ calculation',
            'First auto-vigilance system for IIT metrics'
        ],
        
        'advantages': [
            '500,000-1,000,000× speedup vs naive implementation',
            '50× reduction in corruption-related failures',
            '5-15% accuracy improvement with domain preprocessing',
            'Real-time capability for systems up to N=200 components'
        ],
        
        'applications': [
            'Autonomous aerial vehicles (UAVs)',
            'Self-driving cars',
            'Medical monitoring devices',
            'Smart city infrastructure'
        ],
        
        'markets': [
            'Autonomous systems ($70B+ TAM)',
            'Aerospace & defense ($2B)',
            'Automotive ($50B)',
            'Healthcare ($5B)',
            'Smart cities ($15B)'
        ],
        
        'files_to_protect': [
            'phi_calculator_enhanced_v2.py',
            'SCIENTIFIC_PAPER.md',
            'universal_consciousness_framework.py',
            'TUCHOS_INTEGRATION_ANALYSIS.md'
        ]
    }
    
    print("📋 Configuration:")
    print(f"   Inventor: {config['inventor_name']}")
    print(f"   Title: {config['invention_title']}")
    print(f"   Files to protect: {len(config['files_to_protect'])}")
    print()
    
    # Create invention document
    print("🔨 Creating invention disclosure document...")
    invention = create_invention_document(config)
    
    # Save document
    output_path = 'INVENTION_DISCLOSURE.json'
    save_invention_document(invention, output_path)
    
    # Calculate document hash
    doc_hash = calculate_document_hash(output_path)
    print(f"📊 Document hash (SHA-256): {doc_hash}")
    
    # Prepare list of files to timestamp
    files_to_timestamp = [output_path] + config['files_to_protect']
    
    # Generate OpenTimestamps commands
    generate_opentimestamps_commands(files_to_timestamp)
    
    # Generate Ethereum example
    generate_ethereum_example(doc_hash)
    
    # Final instructions
    print("\n" + "="*70)
    print("✅ NEXT STEPS")
    print("="*70)
    print("""
1. IMMEDIATE (Today):
   - Run OpenTimestamps commands above (FREE)
   - Store .ots files in multiple safe locations
   - Keep original files unchanged

2. SECURE STORAGE:
   - Cloud backup (encrypted): Dropbox, Google Drive
   - Physical backup: USB drive in safe/safety deposit box
   - Version control: Private GitLab/GitHub repo

3. DOCUMENTATION:
   - Print this invention document
   - Have witness sign/date physical copy
   - Notarize if possible (adds legal weight)

4. ONGOING:
   - Update timestamps for new versions
   - Monitor competition (Google alerts, patent searches)
   - Maintain as trade secret until ready to patent

5. LEGAL PROTECTION:
   - Use NDAs before showing to anyone
   - Code obfuscation for distribution (PyArmor)
   - Consider provisional patent when funded ($2K-5K)

REMEMBER:
✅ Blockchain timestamp = Proof of invention date
✅ Trade secret = Immediate protection
✅ Patent later = When you have resources + position

COST TODAY: $0 (OpenTimestamps)
PROTECTION: Perpetual (blockchain immutable)
TIME: 1-2 hours total
""")
    
    print("="*70)
    print("🎉 IP PROTECTION SETUP COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
