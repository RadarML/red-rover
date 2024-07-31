Current configurations (output of `./env/bin/python config/summary.py`):

```
dart.yaml         // DART, indoor
-------------------------------------------
Antenna       2tx x 4rx
Range         4.4cm   x 128 (rmax=5.59m)
Roppler       1.9cm/s x 128 (dmax=1.21m/s)
Throughput    42mbps / 701mbps
Duty cycle    99.8%
Excess ramp   2.100us
```

```
radarhd-b.yaml    // RadarHD, bike-mounted
-------------------------------------------
Antenna       3tx x 4rx
Range         17.5cm   x 256 (rmax=44.74m)
Roppler       23.4cm/s x  64 (dmax=7.49m/s)
Throughput    126mbps / 701mbps
Duty cycle    16.5%
Excess ramp   2.700us
```

```
radarhd-i.yaml    // RadarHD, indoor
-------------------------------------------
Antenna       3tx x 4rx
Range         4.4cm   x 256 (rmax=11.18m)
Roppler       3.8cm/s x  64 (dmax=1.22m/s)
Throughput    126mbps / 701mbps
Duty cycle    99.5%
Excess ramp   2.100us
```

```
radarhd-o.yaml    // RadarHD, outdoor
-------------------------------------------
Antenna       3tx x 4rx
Range         8.7cm   x 256 (rmax=22.37m)
Roppler       3.9cm/s x  64 (dmax=1.24m/s)
Throughput    126mbps / 701mbps
Duty cycle    99.5%
Excess ramp   2.100us
```
