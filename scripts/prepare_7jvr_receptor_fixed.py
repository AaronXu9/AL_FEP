#!/usr/bin/env python3
"""
Prepare 7JVR receptor for AutoDock Vina docking (Fixed version)
This script properly prepares the receptor without ligand-like ROOT/ENDROOT tags.
"""

import subprocess
import os
from pathlib import Path


def prepare_receptor():
    """Prepare the 7JVR receptor file for Vina docking."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    pdb_file = project_root / "data/targets/7jvr/7JVR.pdb"
    output_file = project_root / "data/targets/7jvr/7jvr_prepared_fixed.pdbqt"
    
    print(f"🔬 Preparing 7JVR receptor for AutoDock Vina")
    print(f"Input: {pdb_file}")
    print(f"Output: {output_file}")
    
    if not pdb_file.exists():
        print(f"❌ PDB file not found: {pdb_file}")
        return False
    
    # Method 1: Use obabel with specific options to avoid ROOT/ENDROOT tags
    print(f"\n📝 Converting PDB to PDBQT (rigid receptor)...")
    
    try:
        # Convert without flexibility (rigid receptor)
        cmd = [
            "obabel",
            str(pdb_file),
            "-O", str(output_file),
            "-xr"  # Remove all hydrogens and add only polar ones (for rigid receptor)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"✅ Receptor prepared successfully!")
            print(f"   - Output file: {output_file}")
            print(f"   - File size: {file_size / 1024:.1f} KB")
            
            # Check if there are ROOT/ENDROOT tags (there shouldn't be)
            with open(output_file, 'r') as f:
                content = f.read()
                if "ROOT" in content or "ENDROOT" in content:
                    print(f"⚠️  Warning: ROOT/ENDROOT tags found - cleaning up...")
                    
                    # Remove ROOT/ENDROOT lines for rigid receptor
                    lines = content.split('\n')
                    cleaned_lines = []
                    skip_until_endroot = False
                    
                    for line in lines:
                        if line.strip() == "ROOT":
                            skip_until_endroot = True
                            continue
                        elif line.strip() == "ENDROOT":
                            skip_until_endroot = False
                            continue
                        elif not skip_until_endroot:
                            cleaned_lines.append(line)
                    
                    # Write cleaned content
                    with open(output_file, 'w') as f_out:
                        f_out.write('\n'.join(cleaned_lines))
                    
                    print(f"✅ Cleaned receptor file (removed ROOT/ENDROOT tags)")
                else:
                    print(f"✅ No ROOT/ENDROOT tags found - receptor is properly rigid")
            
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"❌ OpenBabel conversion failed: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = prepare_receptor()
    if success:
        print(f"\n🎉 Receptor preparation completed successfully!")
        print(f"✅ Ready for AutoDock Vina docking")
    else:
        print(f"\n❌ Receptor preparation failed")
