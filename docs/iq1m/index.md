# I/Q-1M: one million i/q frames

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 5px;">
    <img src="images/iq1m-bike.jpg" alt="IQ1M Bike" style="width: 100%; min-width: 200px;">
    <img src="images/iq1m-indoor.jpg" alt="IQ1M Indoor" style="width: 100%; min-width: 200px;">
    <img src="images/iq1m-outdoor.jpg" alt="IQ1M Outdoor" style="width: 100%; min-width: 200px;">
</div>

!!! info

    We are currently working to prepare the dataset for public release and distribution. For the time being, please contact Tianshu Huang (<tianshu2@andrew.cmu.edu>) for access.

The I/Q-1M dataset consists of 1M radar-lidar-camera samples[^1] over 29 hours across indoor, outdoor, and bike-mounted settings, each with a mobile observer:

- `indoor`: inside buildings at a slow to moderate walking pace, visiting multiple floors and areas within each.
- `outdoor`: neighborhoods ranging from single family detached to high density commercial zoning at a moderate to fast walking pace.
- `bike`: bike rides in different directions from a set starting point with a moderate biking pace.

!!! tip

    See our paper for more details about the dataset. Make sure to download the arxiv version to see the attached (and linked) appendix!

[^1]: The radar was collected at 20Hz, the Lidar at 10Hz, and the camera at 30Hz; as such, Lidar is limiting sensor to arrive at our 1M sample count.

| Setting | Size | Length | Average Speed | Max Doppler | Max Range |
|---------|------|--------|---------------|-------------|-----------|
| `indoor` | 310k | 8.9h | 1.0m/s | 1.2m/s | 11.2m |
| `outdoor` | 372k | 10.7h | 1.4m/s | 1.8m/s | 22.4m |
| `bike` | 333k | 9.3h | 5.4m/s | 8.0m/s | 22.4m |
