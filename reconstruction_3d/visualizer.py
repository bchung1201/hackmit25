"""
3D Scene Visualization Module
Creates interactive 3D visualizations of reconstructed scenes
"""

import numpy as np
import open3d as o3d
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json

from .scene_assembler import Scene3D
from .room_reconstructor import Room3DModel
from .object_reconstructor import Object3DModel

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for 3D visualization"""
    window_width: int = 1200
    window_height: int = 800
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    point_size: float = 2.0
    line_width: float = 2.0
    show_axes: bool = True
    show_room_labels: bool = True
    show_furniture_labels: bool = True
    enable_interaction: bool = True

class SceneVisualizer:
    """3D scene visualization using Open3D"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.vis = None
        logger.info("Scene visualizer initialized")
    
    def visualize_scene(self, scene: Scene3D, 
                       window_name: str = "3D Scene Reconstruction") -> bool:
        """
        Visualize complete 3D scene
        
        Args:
            scene: Scene3D to visualize
            window_name: Name of the visualization window
            
        Returns:
            True if visualization was successful
        """
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name=window_name,
                width=self.config.window_width,
                height=self.config.window_height
            )
            
            # Set background color
            opt = self.vis.get_render_option()
            opt.background_color = np.array(self.config.background_color)
            opt.point_size = self.config.point_size
            opt.line_width = self.config.line_width
            
            # Add room geometries
            self._add_rooms_to_visualizer(scene.rooms)
            
            # Add furniture geometries
            self._add_furniture_to_visualizer(scene.furniture)
            
            # Add coordinate axes
            if self.config.show_axes:
                self._add_coordinate_axes()
            
            # Add room labels
            if self.config.show_room_labels:
                self._add_room_labels(scene.rooms)
            
            # Add furniture labels
            if self.config.show_furniture_labels:
                self._add_furniture_labels(scene.furniture)
            
            # Set up camera
            self._setup_camera(scene)
            
            # Run visualization
            if self.config.enable_interaction:
                self.vis.run()
            else:
                # Just capture and close
                self.vis.poll_events()
                self.vis.update_renderer()
            
            return True
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return False
        
        finally:
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None
    
    def _add_rooms_to_visualizer(self, rooms: Dict[str, Room3DModel]):
        """Add room geometries to visualizer"""
        colors = self._get_room_colors(len(rooms))
        
        for i, (room_name, room) in enumerate(rooms.items()):
            if len(room.point_cloud.points) > 0:
                # Color the point cloud
                room_pcd = room.point_cloud
                room_pcd.paint_uniform_color(colors[i % len(colors)])
                
                # Add point cloud
                self.vis.add_geometry(room_pcd)
                
                # Add mesh if available
                if room.mesh is not None and len(room.mesh.vertices) > 0:
                    room_mesh = room.mesh
                    room_mesh.paint_uniform_color(colors[i % len(colors)])
                    self.vis.add_geometry(room_mesh)
    
    def _add_furniture_to_visualizer(self, furniture: List[Object3DModel]):
        """Add furniture geometries to visualizer"""
        colors = self._get_furniture_colors()
        
        for obj in furniture:
            if len(obj.point_cloud.points) > 0:
                # Get color for furniture class
                color = colors.get(obj.class_name, (0.8, 0.8, 0.8))
                
                # Color the point cloud
                obj_pcd = obj.point_cloud
                obj_pcd.paint_uniform_color(color)
                
                # Add point cloud
                self.vis.add_geometry(obj_pcd)
                
                # Add mesh if available
                if obj.mesh is not None and len(obj.mesh.vertices) > 0:
                    obj_mesh = obj.mesh
                    obj_mesh.paint_uniform_color(color)
                    self.vis.add_geometry(obj_mesh)
    
    def _add_coordinate_axes(self):
        """Add coordinate axes to visualizer"""
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(coord_frame)
    
    def _add_room_labels(self, rooms: Dict[str, Room3DModel]):
        """Add room name labels to visualizer"""
        for room_name, room in rooms.items():
            if room.room_center is not None:
                # Create text label (simplified - Open3D doesn't have built-in text)
                # This would require additional text rendering implementation
                pass
    
    def _add_furniture_labels(self, furniture: List[Object3DModel]):
        """Add furniture labels to visualizer"""
        for obj in furniture:
            if obj.center is not None:
                # Create text label (simplified - Open3D doesn't have built-in text)
                # This would require additional text rendering implementation
                pass
    
    def _setup_camera(self, scene: Scene3D):
        """Set up camera view for the scene"""
        if scene.scene_center is not None and scene.scene_dimensions is not None:
            # Calculate camera position
            center = scene.scene_center
            dimensions = scene.scene_dimensions
            
            # Position camera at a good distance
            max_dim = np.max(dimensions)
            camera_distance = max_dim * 2.0
            
            # Set camera position
            ctr = self.vis.get_view_control()
            ctr.set_lookat(center)
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            ctr.set_zoom(0.8)
    
    def _get_room_colors(self, num_rooms: int) -> List[Tuple[float, float, float]]:
        """Get distinct colors for rooms"""
        colors = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
            (0.5, 0.5, 0.5),  # Gray
            (0.8, 0.5, 0.2),  # Orange
        ]
        
        # Repeat colors if needed
        return colors * ((num_rooms // len(colors)) + 1)
    
    def _get_furniture_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Get colors for different furniture types"""
        return {
            'chair': (0.6, 0.4, 0.2),      # Brown
            'couch': (0.4, 0.2, 0.6),      # Purple
            'bed': (0.2, 0.6, 0.4),        # Green
            'dining table': (0.6, 0.6, 0.2), # Yellow
            'tv': (0.2, 0.2, 0.2),         # Dark gray
            'refrigerator': (0.8, 0.8, 0.8), # Light gray
            'microwave': (0.4, 0.4, 0.4),   # Gray
            'oven': (0.3, 0.3, 0.3),       # Dark gray
            'sink': (0.7, 0.7, 0.9),       # Light blue
            'toilet': (0.9, 0.9, 0.9),     # White
        }
    
    def capture_screenshot(self, scene: Scene3D, output_path: str) -> bool:
        """
        Capture screenshot of the scene
        
        Args:
            scene: Scene3D to capture
            output_path: Path to save screenshot
            
        Returns:
            True if successful
        """
        try:
            # Create visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name="Screenshot",
                width=self.config.window_width,
                height=self.config.window_height,
                visible=False  # Don't show window
            )
            
            # Set background color
            opt = self.vis.get_render_option()
            opt.background_color = np.array(self.config.background_color)
            opt.point_size = self.config.point_size
            opt.line_width = self.config.line_width
            
            # Add geometries
            self._add_rooms_to_visualizer(scene.rooms)
            self._add_furniture_to_visualizer(scene.furniture)
            
            if self.config.show_axes:
                self._add_coordinate_axes()
            
            # Set up camera
            self._setup_camera(scene)
            
            # Capture image
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # Save screenshot
            self.vis.capture_screen_image(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return False
        
        finally:
            if self.vis is not None:
                self.vis.destroy_window()
                self.vis = None
    
    def generate_html_viewer(self, scene: Scene3D, output_path: str) -> bool:
        """
        Generate HTML viewer for the scene
        
        Args:
            scene: Scene3D to visualize
            output_path: Path to save HTML file
            
        Returns:
            True if successful
        """
        try:
            # Create combined point cloud
            from .scene_assembler import SceneAssembler
            assembler = SceneAssembler()
            combined_pcd = assembler.create_combined_point_cloud(scene)
            
            if len(combined_pcd.points) == 0:
                logger.warning("No points to visualize")
                return False
            
            # Convert to PLY format
            ply_path = output_path.replace('.html', '.ply')
            o3d.io.write_point_cloud(ply_path, combined_pcd)
            
            # Generate HTML viewer
            html_content = self._generate_html_content(ply_path, scene)
            
            # Save HTML file
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML viewer generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"HTML viewer generation failed: {e}")
            return False
    
    def _generate_html_content(self, ply_path: str, scene: Scene3D) -> str:
        """Generate HTML content for 3D viewer"""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Scene Reconstruction</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background: #111; }}
        #container {{ width: 100vw; height: 100vh; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; font-family: Arial; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h3>3D Scene Reconstruction</h3>
        <p>Rooms: {len(scene.rooms)}</p>
        <p>Furniture: {len(scene.furniture)}</p>
        <p>Controls: Mouse to rotate, scroll to zoom</p>
    </div>
    
    <script>
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('container').appendChild(renderer.domElement);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Load PLY file
        const loader = new THREE.PLYLoader();
        loader.load('{ply_path}', function(geometry) {{
            const material = new THREE.PointsMaterial({{ 
                size: 0.01,
                vertexColors: true 
            }});
            const points = new THREE.Points(geometry, material);
            scene.add(points);
            
            // Center camera
            const box = new THREE.Box3().setFromObject(points);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            camera.position.set(center.x, center.y, cameraZ);
            controls.target.copy(center);
        }});
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Handle window resize
        window.addEventListener('resize', function() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
    
    def export_scene_data(self, scene: Scene3D, output_path: str) -> bool:
        """
        Export scene data for external viewers
        
        Args:
            scene: Scene3D to export
            output_path: Directory to save exported files
            
        Returns:
            True if successful
        """
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export scene statistics
            from .scene_assembler import SceneAssembler
            assembler = SceneAssembler()
            stats = assembler.get_scene_statistics(scene)
            
            stats_path = output_dir / "scene_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            # Export room data
            rooms_dir = output_dir / "rooms"
            rooms_dir.mkdir(exist_ok=True)
            
            for room_name, room in scene.rooms.items():
                room_data = {
                    'name': room_name,
                    'center': room.room_center.tolist() if room.room_center is not None else None,
                    'dimensions': room.room_dimensions.tolist() if room.room_dimensions is not None else None,
                    'point_count': len(room.point_cloud.points),
                    'has_mesh': room.mesh is not None
                }
                
                room_file = rooms_dir / f"{room_name}.json"
                with open(room_file, 'w') as f:
                    json.dump(room_data, f, indent=2)
            
            # Export furniture data
            furniture_dir = output_dir / "furniture"
            furniture_dir.mkdir(exist_ok=True)
            
            for i, obj in enumerate(scene.furniture):
                obj_data = {
                    'id': obj.object_id,
                    'class': obj.class_name,
                    'center': obj.center.tolist() if obj.center is not None else None,
                    'dimensions': obj.dimensions.tolist() if obj.dimensions is not None else None,
                    'confidence': obj.confidence,
                    'point_count': len(obj.point_cloud.points),
                    'has_mesh': obj.mesh is not None
                }
                
                obj_file = furniture_dir / f"{obj.class_name}_{i}.json"
                with open(obj_file, 'w') as f:
                    json.dump(obj_data, f, indent=2)
            
            logger.info(f"Scene data exported to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Scene data export failed: {e}")
            return False
