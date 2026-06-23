import pyvista as pv

def render_gltf(gltf_file, out_prefix):
    # Load GLTF
    plotter = pv.Plotter(off_screen=True)
    
    # Read the block
    block = pv.read(gltf_file)
    plotter.add_mesh(block, smooth_shading=False)
    
    # We want 6 views: 3 orthogonal, 3 isometric
    # For a box 100x100x20
    
    views = [
        ("ortho_top", (0, 0, 1), (0, 1, 0)),    # camera pos, view up
        ("ortho_front", (0, -1, 0), (0, 0, 1)),
        ("ortho_right", (1, 0, 0), (0, 0, 1)),
        ("iso_1", (1, 1, 1), (0, 0, 1)),
        ("iso_2", (-1, -1, 1), (0, 0, 1)),
        ("iso_3", (1, -1, 1), (0, 0, 1))
    ]
    
    for name, pos, viewup in views:
        plotter.camera_position = [pos, (0,0,0), viewup]
        plotter.camera.SetParallelProjection(True) # Optional, parallel is often better for CAD
        plotter.reset_camera()
        
        # Add some lighting
        plotter.enable_lightkit()
        plotter.set_background('white')
        
        plotter.screenshot(f"{out_prefix}_{name}.png")

if __name__ == "__main__":
    render_gltf("test.gltf", "test")
