include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "imu_link",
  published_frame = "radar",
  odom_frame = "base_link",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.0,
  publish_tracked_pose = true,
}


-- Tuning: see
-- [1] Cartographer-ROS algorithm walkthrough
--     https://google-cartographer-ros.readthedocs.io/en/latest/algo_walkthrough.html
-- [2] Cartographer configuration documentation
--     https://google-cartographer.readthedocs.io/en/latest/configuration.html
-- [3] Cartographer default configuration values
--     (note: tuned for google street view backpack -- outdoors)
--     https://github.com/cartographer-project/cartographer/blob/master/configuration_files/trajectory_builder_3d.lua
--     https://github.com/cartographer-project/cartographer/blob/master/configuration_files/map_builder.lua
--     https://github.com/cartographer-project/cartographer/blob/master/configuration_files/pose_graph.lua


-- TRAJECTORY_BUILDER [3.1]

-- Sensor carrier exclusion: 0.5m; no real maximum
TRAJECTORY_BUILDER_3D.min_range = 0.5
TRAJECTORY_BUILDER_3D.max_range = 20

-- 1 message per scan
-- The entire pipeline is very slow when each column is provided separately.
TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1

-- Scans per local submap; each local submap is assumed to be locally correct.
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 100

-- Not sure what this does; see [2].
TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching = false

-- Cartographer first downsamples point clouds into voxels targeting a fixed
-- number of points (voxel centers).
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 1.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 250.
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 2.
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 400.

-- Cartographer then tries to align the low res voxels to the low res submap,
-- followed by tuning the high res voxels on the high res submap.
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.05
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.1

-- Give ceres a lot of iterations to make sure it will always converge
-- We're also not trying to run real time anyways
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 500

-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 1.0
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 0.1


-- MAP_BUILDER [3.2]

-- 3D slam on all available threads
MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 32


-- POSE_GRAPH [3.3]

-- Optimize the pose graph after adding every `n` nodes.
-- Set to 0 to disable global loop closure. Otherwise, the value shouldn't
-- make a significant impact on offline slam.
POSE_GRAPH.optimize_every_n_nodes = 50

-- Give ceres a lot of iterations to make sure it will always converge
-- We're also not trying to run real time anyways
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 500
POSE_GRAPH.constraint_builder.ceres_scan_matcher.ceres_solver_options.max_num_iterations = 500

-- Minimum score needed to make a constraint between nearby nodes.
-- Lower = more constraints, but with a higher chance of erroneous matches.
POSE_GRAPH.constraint_builder.min_score = 0.6

-- Minimum score needed ot make a loop closure constraint in the global map.
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6

-- Higher huber scale = more impactful outliers.
-- Tend to require a higher value if more "loops" are taken.
POSE_GRAPH.optimization_problem.huber_scale = 5e2

-- Sample a subset of nodes to build constraints.
POSE_GRAPH.constraint_builder.sampling_ratio = 1.0

-- Maximum number of cleanup iterations to run.
POSE_GRAPH.max_num_final_iterations = 200000

-- Print optimization progress metadata.
POSE_GRAPH.optimization_problem.log_solver_summary = true

return options
