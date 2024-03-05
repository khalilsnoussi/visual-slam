import open3d as o3d
import numpy as np
from slam import Frame, SLAM



#creating a list of Frame objects
slam= SLAM()
slam.feature_extractor()


# Create a visualizer object

#vis = o3d.visualization.Visualizer()
#vis.create_window()
#frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin = [0,0,0])
#vis.add_geometry(frame_mesh)

# Visualize camera poses with 3D frames
for pose in slam.poses:
    pose = (np.hstack((pose[0], pose[1])))
    pose = pose.flatten()
    pose = pose.tolist()
    with open('poses.txt', 'a') as file:
        file.write(str(pose) + '\n')
    # pose is a (R,t) like tuple
    #new_frame = frame_mesh.transform(pose)
    #vis.add_geometry(new_frame)
   
    


#render_options = vis.get_render_option()
#render_options.point_size = 2


#vis.run()
#vis.destroy_window()


