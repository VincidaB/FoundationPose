# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse
from time import time

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/ipd_val_0/mesh/obj_000018.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/ipd_val_0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  parser.add_argument('--profile', type=bool, default=False)
  args = parser.parse_args()

  set_logging_format(level=logging.ERROR)
  set_seed(0)

  mesh = trimesh.load(args.mesh_file)
  # scale mesh to m from mm
  mesh.apply_transform(np.diag([0.001,0.001,0.001,1]))

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  # logging.info("estimator initialization done")

  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  for i in range(len(reader.color_files)):
    # logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    # scale depth by a factor of 0.1
    depth = depth*0.1
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      start_time = time()
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      logging.getLogger().setLevel(logging.INFO)
      logging.info(f'\033[93mregister time first run: {time()-start_time:.2f}\033[0m')
      logging.getLogger().setLevel(logging.ERROR)


      gt_rotation = np.array([-0.07963122209137952, -0.0751164820287667, -0.9939901322154824, 0.9916181604683285, 0.09581080499935263, -0.08668168189401496, 0.10174621806543538, -0.9925612348360231, 0.06685733667636427]).reshape(3,3)
      gt_pose_t =np.array([-100.31353924053144, 55.82813669281825, 1702.1111610000928]) /1000
      # logging.info(f'gt_pose_t: {gt_pose_t}')

      # calculate the error in position
      estimated_position = pose[:3,3]
      # logging.info(f'Estimated position: {estimated_position}')
      gt_position = gt_pose_t
      error = np.linalg.norm(estimated_position - gt_position)

      # calculate the error in rotation
      estimated_rotation = pose[:3,:3]

      def getAngle(P, Q):
        R = Q @ P.T
        cos_theta = (np.trace(R)-1)/2
        return np.arccos(cos_theta) * 180 / np.pi
      
      angle = getAngle(estimated_rotation, gt_rotation)
      print(f'\033[92m{"="*40}\033[0m')
      print(f'\033[92mError in position: {error:.4f}m\033[0m')
      print(f'\033[92mAngle est/gt rotation: {angle:.2f} deg\033[0m')
      print(f'\033[92m{"="*40}\033[0m')

      # second run for performance evaluation
      start_time = time()
      #disable logging for second run
      logging.getLogger().setLevel(logging.ERROR)
      
      profile = args.profile
      if profile:
        from cProfile import Profile
        from pstats import SortKey, Stats
        import io
        profiler = Profile()
        profiler.enable()  # this line can be wherever we want to start collecting data
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      if profile:
        profiler.disable()  # this line can be wherever we want to stop collecting data
        p_stream = io.StringIO()
        sortby = SortKey.NFL
        stats = Stats(profiler, stream=p_stream)
        # stats.strip_dirs()
        stats.sort_stats(sortby)
        stats.print_stats(
          # 0.1,  # top 10% of lines
          "FoundationPose",  # only display functions from this directory
        )
        # Process the output to remove everything before "FoundationPose"
        output = p_stream.getvalue()
        processed_output = re.sub(r'/home/.*/FoundationPose', ' .../FoundationPose', output)
        print(processed_output)
      logging.getLogger().setLevel(logging.INFO)
      logging.info(f'\033[93mregister time second run: {time()-start_time:.2f}\033[0m')
      logging.getLogger().setLevel(logging.ERROR)

      # third run for performance evaluation
      start_time = time()
      #disable logging for third run
      logging.getLogger().setLevel(logging.ERROR)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
      logging.getLogger().setLevel(logging.INFO)
      logging.info(f'\033[93mregister time third run: {time()-start_time:.2f}\033[0m')
      logging.getLogger().setLevel(logging.ERROR)

      if debug>=3:
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)
