"""Summarize available configurations."""

import os, yaml
import sys

if sys.path[0] != '':
    sys.path.insert(0, '')

from rover import radar_api


def _print_cfg(name):
    with open(os.path.join("config", name)) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    radar = radar_api.RadarConfig(**cfg["radar"]["args"]["radar"])
    capture = radar_api.CaptureConfig(**cfg["radar"]["args"]["capture"])
    intrinsics = radar.as_intrinsics()
    rr = intrinsics['range_resolution']
    nd, tx, rx, nr = intrinsics['shape']
    rd = intrinsics['doppler_resolution']

    print(f"""
{name:16}  // {cfg['radar']['comment']}
-------------------------------------------
Antenna       {tx}tx x {rx}rx
Range         {(rr * 100):.1f}cm   x {nr:3} (rmax={(rr * nr):.2f}m)
Roppler       {(rd * 100):.1f}cm/s x {nd:3} (dmax={(rd * nd / 2):.2f}m/s)
Throughput    {(radar.throughput / 1e6):.0f}mbps / {(capture.throughput / 1e6):.0f}mbps
Duty cycle    {(radar.frame_time / radar.frame_period * 100):.1f}%
Excess ramp   {(radar.ramp_end_time - radar.adc_start_time - radar.sample_time):.3f}us
""")


for name in sorted(os.listdir("config")):
    if name.endswith('.yaml'):
        _print_cfg(name)
