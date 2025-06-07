"""Roverd: data format and data loading.

!!! tip "ADL-Compliant"

    The `roverd` package implements a fully [Abstract Dataloader](
    https://wiselabcmu.github.io/abstract-dataloader/)-compliant map-style
    data loader.

    Thus, to use the dataloader in practice, in addition to writing custom
    ADL-compliant components, you can use generic ADL components:

    - [`Nearest`][abstract_dataloader.generic.Nearest] synchronization
    - [`Window`][abstract_dataloader.generic.Window] to load consecutive frames
        as a single sample
    - [`TransformedDataset`][abstract_dataloader.torch.TransformedDataset] to
        get the [`roverd.Dataset`][roverd.Dataset] into a pytorch dataloader

!!! success "Fully Typed"

    The `roverd` dataloader is fully typed using generic dataclasses of
    [jaxtyping](https://github.com/patrick-kidger/jaxtyping) arrays following
    the [Abstract Dataloader's recommendations](
    https://wiselabcmu.github.io/abstract-dataloader/types/), and comes with a
    [type library](types.md) which describes the data types collected by the
    `red-rover` system.

To use `roverd`, you can either use the high level interfaces to load a
complete dataset consisting of multiple traces, or use lower-level APIs
to load a single trace, single sensor, or a single "channel" within a sensor.

- High level APIs are generally preferred, and include descriptive types.
- Lower level APIs should generally be avoided, but are required to
    *modify* the data; high-level APIs are intentionally read only.

=== "Dataset"

    ```python
    >>> from roverd import Dataset, sensors
    >>> from abstract_dataloader import generic
    >>> dataset = Dataset.from_config(
            Dataset.find_traces("/data/grt"),
            sync=generic.Nearest("lidar", tol=0.1),
            sensors={"radar": sensors.XWRRadar, "lidar": sensors.OS0Lidar})
    >>> dataset
    Dataset(166 traces, n=1139028)
    >>> dataset[42]
    {'radar': XWRRadarIQ(...), 'lidar': OS0LidarData(...)}
    ```

=== "Single Trace, Typed"

    ```python
    >>> from roverd import Trace, sensors
    >>> from abstract_dataloader import generic
    >>> trace = Trace.from_config(
            "/data/grt/bike/point.back",
            sync=generic.Nearest("lidar"),
            sensors={"radar": sensors.XWRRadar, "lidar": sensors.OS0Lidar})
    >>> trace['radar']
    XWRRadar(/data/grt/bike/point.back/radar: [ts, iq, valid])
    >>> trace[42]
    {'radar': XWRRadarIQ(...), 'lidar': OS0LidarData(...)}
    ```

=== "Single Trace, Untyped"

    ```python
    >>> from roverd import Trace
    >>> from abstract_dataloader import generic
    >>> trace = Trace.from_config(
            "/data/grt/bike/point.back", sync=generic.Nearest("lidar"))
    >>> trace
    Trace(/data/grt/bike/point.back, 12195x[radar, camera, lidar, imu])
    >>> trace['radar']
    DynamicSensor(/data/grt/bike/point.back/radar: [ts, iq, valid])
    >>> trace[42]
    {'radar': {...}, 'camera': {...}, 'lidar': {...}, ...}
    ```

=== "Single Sensor"

    ```python
    >>> from roverd.sensors import XWRRadar
    >>> radar = XWRRadar("/data/grt/bike/point.back/radar")
    >>> radar
    XWRRadar(/data/grt/bike/point.back/radar: [ts, iq, valid])
    >>> len(radar)
    24576
    >>> radar[42]
    XWRRadarIQ(
        iq=int16[1 64 3 4 512], timestamps=float64[1], valid=uint8[1],
        range_resolution=float32[1], doppler_resolution=float32[1])
    ```
"""

from jaxtyping import install_import_hook

with install_import_hook("roverd", "beartype.beartype"):
    from . import channels, sensors, timestamps, types
    from .dataset import Dataset, Trace

__all__ = ["channels", "timestamps", "sensors", "types", "Dataset", "Trace"]
