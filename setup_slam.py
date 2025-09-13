#!/usr/bin/env python3
"""
SLAM Integration Setup Script
Automates the setup of advanced Gaussian Splatting SLAM systems
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SLAMSetup:
    """SLAM integration setup manager"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.external_dir = self.project_root / "external"
        
        # SLAM repositories to clone
        self.slam_repos = {
            'SplaTAM': 'https://github.com/spla-tam/SplaTAM.git',
            'MonoGS': 'https://github.com/muskie82/MonoGS.git',
            'Splat-SLAM': 'https://github.com/eriksandstroem/Splat-SLAM.git',
            'Gaussian-SLAM': 'https://github.com/VladimirYugay/Gaussian-SLAM.git'
        }
    
    async def setup_all(self):
        """Run complete SLAM setup"""
        logger.info("🚀 Starting SLAM Integration Setup")
        logger.info("=" * 50)
        
        try:
            # Step 1: Create directories
            await self.create_directories()
            
            # Step 2: Install Python dependencies
            await self.install_dependencies()
            
            # Step 3: Clone SLAM repositories
            await self.clone_slam_repos()
            
            # Step 4: Setup SLAM backends
            await self.setup_slam_backends()
            
            # Step 5: Run tests
            await self.run_tests()
            
            logger.info("✅ SLAM Integration Setup Complete!")
            await self.print_usage_instructions()
            
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise
    
    async def create_directories(self):
        """Create necessary directories"""
        logger.info("📁 Creating directories...")
        
        directories = [
            self.external_dir,
            self.project_root / "logs",
            self.project_root / "data",
            self.project_root / "models",
            self.project_root / "outputs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"   Created: {directory}")
    
    async def install_dependencies(self):
        """Install Python dependencies"""
        logger.info("📦 Installing Python dependencies...")
        
        # Install basic requirements
        await self.run_command([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        # Install SLAM-specific requirements
        slam_requirements = self.project_root / "requirements_slam.txt"
        if slam_requirements.exists():
            await self.run_command([
                sys.executable, "-m", "pip", "install", "-r", str(slam_requirements)
            ])
        
        logger.info("✅ Dependencies installed")
    
    async def clone_slam_repos(self):
        """Clone SLAM repositories"""
        logger.info("📥 Cloning SLAM repositories...")
        
        for repo_name, repo_url in self.slam_repos.items():
            repo_path = self.external_dir / repo_name
            
            if repo_path.exists():
                logger.info(f"   {repo_name} already exists, skipping...")
                continue
            
            logger.info(f"   Cloning {repo_name}...")
            try:
                await self.run_command([
                    "git", "clone", "--recursive", repo_url, str(repo_path)
                ])
                logger.info(f"   ✅ {repo_name} cloned successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"   ⚠️  Failed to clone {repo_name}: {e}")
                logger.info(f"   You can manually clone from: {repo_url}")
    
    async def setup_slam_backends(self):
        """Setup individual SLAM backends"""
        logger.info("⚙️  Setting up SLAM backends...")
        
        # Setup SplaTAM
        await self.setup_splatam()
        
        # Setup MonoGS
        await self.setup_monogs()
        
        # Setup Splat-SLAM
        await self.setup_splat_slam()
        
        # Setup Gaussian-SLAM
        await self.setup_gaussian_slam()
    
    async def setup_splatam(self):
        """Setup SplaTAM backend"""
        logger.info("🔧 Setting up SplaTAM...")
        
        splatam_path = self.external_dir / "SplaTAM"
        if not splatam_path.exists():
            logger.warning("   SplaTAM not found, skipping setup")
            return
        
        try:
            # Install SplaTAM dependencies
            requirements_file = splatam_path / "requirements.txt"
            if requirements_file.exists():
                await self.run_command([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], cwd=splatam_path)
            
            # Build any necessary components
            setup_py = splatam_path / "setup.py"
            if setup_py.exists():
                await self.run_command([
                    sys.executable, "setup.py", "build_ext", "--inplace"
                ], cwd=splatam_path)
            
            logger.info("   ✅ SplaTAM setup complete")
            
        except Exception as e:
            logger.warning(f"   ⚠️  SplaTAM setup failed: {e}")
    
    async def setup_monogs(self):
        """Setup MonoGS backend"""
        logger.info("🔧 Setting up MonoGS...")
        
        monogs_path = self.external_dir / "MonoGS"
        if not monogs_path.exists():
            logger.warning("   MonoGS not found, skipping setup")
            return
        
        try:
            # Install MonoGS dependencies
            requirements_file = monogs_path / "requirements.txt"
            if requirements_file.exists():
                await self.run_command([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], cwd=monogs_path)
            
            logger.info("   ✅ MonoGS setup complete")
            
        except Exception as e:
            logger.warning(f"   ⚠️  MonoGS setup failed: {e}")
    
    async def setup_splat_slam(self):
        """Setup Splat-SLAM backend"""
        logger.info("🔧 Setting up Splat-SLAM...")
        
        splat_slam_path = self.external_dir / "Splat-SLAM"
        if not splat_slam_path.exists():
            logger.warning("   Splat-SLAM not found, skipping setup")
            return
        
        try:
            # Install dependencies
            requirements_file = splat_slam_path / "requirements.txt"
            if requirements_file.exists():
                await self.run_command([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], cwd=splat_slam_path)
            
            logger.info("   ✅ Splat-SLAM setup complete")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Splat-SLAM setup failed: {e}")
    
    async def setup_gaussian_slam(self):
        """Setup Gaussian-SLAM backend"""
        logger.info("🔧 Setting up Gaussian-SLAM...")
        
        gaussian_slam_path = self.external_dir / "Gaussian-SLAM"
        if not gaussian_slam_path.exists():
            logger.warning("   Gaussian-SLAM not found, skipping setup")
            return
        
        try:
            # Install dependencies
            requirements_file = gaussian_slam_path / "requirements.txt"
            if requirements_file.exists():
                await self.run_command([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ], cwd=gaussian_slam_path)
            
            logger.info("   ✅ Gaussian-SLAM setup complete")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Gaussian-SLAM setup failed: {e}")
    
    async def run_tests(self):
        """Run basic tests to verify setup"""
        logger.info("🧪 Running setup verification tests...")
        
        try:
            # Test imports
            logger.info("   Testing imports...")
            
            # Test core modules
            test_imports = [
                "import torch",
                "import torchvision", 
                "import cv2",
                "import numpy as np",
                "from enhanced_video_pipeline import EnhancedMentraVideoPipeline",
                "from core.pipeline.slam_manager import SLAMManager",
                "from unified_slam_reconstruction import UnifiedSLAMReconstructor"
            ]
            
            for import_statement in test_imports:
                try:
                    exec(import_statement)
                    logger.info(f"   ✅ {import_statement}")
                except ImportError as e:
                    logger.warning(f"   ⚠️  {import_statement} - {e}")
            
            # Test pipeline creation
            logger.info("   Testing pipeline creation...")
            from enhanced_video_pipeline import EnhancedMentraVideoPipeline
            from config import DEFAULT_CONFIG
            
            config = DEFAULT_CONFIG.to_dict()
            pipeline = EnhancedMentraVideoPipeline(config, use_mock_components=True)
            logger.info("   ✅ Pipeline creation successful")
            
            logger.info("✅ Setup verification complete")
            
        except Exception as e:
            logger.error(f"❌ Setup verification failed: {e}")
            raise
    
    async def run_command(self, cmd: List[str], cwd: Path = None) -> str:
        """Run shell command asynchronously"""
        logger.debug(f"Running: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise subprocess.CalledProcessError(process.returncode, cmd, error_msg)
        
        return stdout.decode()
    
    async def print_usage_instructions(self):
        """Print usage instructions"""
        print("\n" + "=" * 60)
        print("🎉 SLAM INTEGRATION SETUP COMPLETE!")
        print("=" * 60)
        print()
        print("📋 NEXT STEPS:")
        print()
        print("1. Run the main pipeline:")
        print("   python main.py")
        print()
        print("2. Try the real-time SLAM demo:")
        print("   python demos/slam_demos/realtime_slam_demo.py")
        print()
        print("3. Benchmark SLAM backends:")
        print("   python demos/slam_demos/benchmark_backends.py")
        print()
        print("4. Visualize SLAM trajectory:")
        print("   python demos/slam_demos/trajectory_visualization.py")
        print()
        print("📖 CONFIGURATION:")
        print()
        print("• Edit config.py to customize SLAM settings")
        print("• SLAM configs are in configs/slam_configs/")
        print("• Set slam_backend to: auto, splatam, monogs, mock")
        print()
        print("🔧 TROUBLESHOOTING:")
        print()
        print("• Check logs/ directory for detailed logs")
        print("• Use mock components for development: use_mock_components=True")
        print("• GPU required for best performance (CUDA recommended)")
        print()
        print("📚 DOCUMENTATION:")
        print()
        print("• SplaTAM: Real-time RGB-D SLAM")
        print("• MonoGS: Fast monocular SLAM (10+ FPS)")
        print("• Splat-SLAM: High-accuracy SLAM")
        print("• Gaussian-SLAM: Photo-realistic SLAM")
        print()
        print("✨ Happy SLAM-ing!")

async def main():
    """Main setup function"""
    setup = SLAMSetup()
    await setup.setup_all()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
