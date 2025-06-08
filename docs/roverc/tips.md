# Collecting Good Data

## Cartographer

**Avoid rapid rotation and acceleration.** We observe that Cartographer tends to perform poorly when the rig rapidly rotates especially. Keep in mind that velocity differentiates the estimated pose, so is an order of magnitude more sensitive to any error.

In DART configuration, also note that the Lidar runs at 20Hz, while the radar requires the trajectory to be resampled to 10Hz, which is at the Nyquist limit. Combined with the high velocity jitter (at least in the velocity domain) which we observe, possibly due to loop closures which do not preserve second order pose continuity, aggressive smoothing is typically required.

**Minimize vertical acceleration.** Since the Lidar is mostly planar, estimating the vertical position of the rig is relatively imprecise. The typical solution to this problem is to use two Lidars, one of which is downwards-facing, but we do not want to require two Lidars on the rig.

**Keep features visible.** Stay conscious of which portions of the Lidar your body is blocking; this can lead to loop closure failures when traversing tight spaces, especially relatively narrow restrictions combined with occlusions and a tight turn. Consider holding the radar to your side or above your head to maintain two-way visibility.

**Keep a low speed.** The scan rate of the Lidar is limited, so fast movement speeds will reduce velocity estimation accuracy. The 1m/s target speed for the DART config seems to work well when combined with 20Hz Lidar scans.


## DART

**Keep your velocity close to, but under, the maximum doppler.** Sections of the trace which exceed the maximum doppler must be discarded due to doppler ambiguity (i.e. "wrapping"). Sections of the trace which reduce below the minimum doppler (typically `0.2 * max_doppler`) are discarded due to excessively wide / degenerate doppler bins.


## Novel View Synthesis

**Minimize clutter.** Clutter cannot be effectively resolved by our radar, and end up being stochastically averaged (e.g. as blurry regions in synthesized frames).

**Keep the validation set within domain.** All regions covered by the validation step (typically the last 20% of the trace) should be within domain. As a suggested pattern, cover above/below chest/waist level during training (in addition to pointing in-plane), then keep the radar in-plane for validation.

**Prefer inside-out scanning.** Since the radar has a 180 degree FOV, the vast majority of the radar images will always cover the background of a scene. As such, any single object within the scene will always be a negligible contributor to NVS performance.

**Emphasize occlusion.** Any baseline method will work fine in an empty box. Try to find scenes with prominent occlusions at the room scale, e.g. two adjacent rooms or spaces with a dividing wall.


## Tomography

**Capture multiple viewing angles.** Some specular surfaces may not show up unless viewed from specific angles. Cover multiple angles, including at multiple heights, to capture these.

As a suggested pattern, try waist/chest level, above the head, and as low as possible.

**Consider avoiding downwards-facing views.** Floors are often highly specular and highly reflective. Too much emphasis on downwards-facing views may lead to the floor being a dominating feature of the scene. For floorplans, XR tomography visualizations, and other "pretty picture"-type datasets, consider de-emphasizing the floor.

**Prefer outside-in scanning.** Outside-in scenes give the opportunity to capture many different angles of all surfaces, giving better tomography quality when the background is cropped out.

**Maximize the scene size.** Errors in DART appear to be size-invariant (e.g. constant spatial resolution error). Larger scenes will therefore have a higher relative/perceived spatial resolution.


## Deep Radar

**Avoid low / stationary speeds.** Low speeds lead to reduced doppler information.

**Keep a level roll angle.** When used for classic RadarHD training, the Bird's Eye View (BEV) is derived from the horizontal plane of the sensor. Maintain a level roll angle where this plane is aligned with the floor (so the BEV matches what a floor plan would look like).

**Keep a consistent velocity.** Provide consistent doppler to the model.

**Clearly separate movement patterns.** Ensure balance between different movement patterns in the train/val/test splits.

As a suggested split for the handheld case, record one trace when walking forward (velocity aligned with sensor) and another when walking sideways (velocity perpendicular with sensor). Note that backwards movement can be obtained from forwards movement using data augmentation (reverse doppler axis), while left and right-handed lateral movement can be similarly obtained (reverse doppler and azimuth axes).

**Do not enter elevators.** This is literally the kidnapped robot problem, not to mention that elevators are kind of boring as far as data goes.

**Maintain depth diversity when possible.** Try to make sure each frame has a mix of close and far objects when possible. In particular, avoid pointing the sensor directly at a wall from a close distance.
