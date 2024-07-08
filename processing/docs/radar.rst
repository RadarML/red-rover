Radar Processing
================

NOTE: these routines are GPU-accelerated using JAX, which incurs significant
call and initialization overhead. As long as you are using `jaxtyping > 0.2.26`,
jax will not be imported until a function which requires it is called.

.. autofunction:: rover.doppler_range_azimuth

.. autofunction:: rover.doppler_range_azimuth_elevation

.. autoclass:: rover.RadarProcessing
   :members:
   :special-members: __call__

.. autoclass:: rover.CFAR
   :members:
   :special-members: __call__

.. autoclass:: rover.AOAEstimation
   :members:
   :special-members: __call__
