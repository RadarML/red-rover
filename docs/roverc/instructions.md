# Data Collection: Step by Step

!!! info

    This guide assumes you have already assembled, installed, and configured `red-rover`. See the [assembly](./assembly.md) and [software setup & sensor configuration](./setup.md) guides if you have not done so already (or this has not been done for you).

## Physical checks

1. Ensure that the protective cover is removed from the radar.
2. The radar, lidar, camera, and computer should be plugged in.

    !!! danger

        The radar and lidar power supplies use the same-sized barrel jack. Ensure that the radar and lidar power cables are not crossed, since **24V power supplied to the Lidar will fry the radar**.

3. The router should be plugged in to the exposed ethernet cable at the backpack end of the cable snake, along with power. The ethernet cable should be connected to the `LAN` port.

4. Plug an external SSD into the data collection rig using one of the USB C ports on the back of the data collection computer. The SSD should fit in the gap between the camera and the rear plate of the data collection rig.

## Power on

1. Turn on the AC battery by pressing the power button, followed by the `AC` button.

2. Insert the battery into the backpack.

    - Plug the power strip with attached power bricks (which may or may not be inside a cable management box) into one of the battery's AC outlets.

    !!! warning

        Ensure that any power buttons are facing outwards (i.e. away from your back) to prevent accidentally pressing the button during data collection.

3. Verify that the Radar, Lidar, and Camera have turned on automatically.

    !!! info "Expected Behavior"

        - Several lights (red and green) should appear on the radar.
        - The lidar should begin to spin after a few seconds; this will be evident as a subtle vibration/hum when running in 10Hz mode and a more aggressive vibration when running at 20Hz. This mode will match what the Lidar was previously configured to use, which may be different from the configuration which you intend to use.
        - The tally light on the front of the camera will turn on. The tally light will always be white (changing its color to red requires access to a special API via SDI in, HDMI, or ethernet which we do not use).

4. Turn on the computer using the power button on the right.

    !!! info "Expected Behavior"
    
        - A white LED under the power button will illuminate when turned on.

## Configure data collection.

1. Connect a display, keyboard, and mouse (or equivalent, e.g. a ["lapdock"](https://www.amazon.com/NexDock-Touchscreen-Wireless-Portable-Compatible/dp/B0CSK2T47Q/)) to the computer. Log in (the computer should be set to auto-log-in without a password).

2. Open a terminal. Select the desired config, configure, and start the server:

    ```
    export ROVER_CFG=dart.yaml
    sudo -E make config
    make
    ```

    !!! warning

        The config file (`dart.yaml` here) must match the physical hardware used:

        - `lidar/interface` and `radar/interface` should match the ethernet interfaces used to communicate with  the lidar and radar respectively (i.e. in `ifconfig`)
        - `lidar/beams` should match the number of physical beams on the lidar (i.e. `beams: 128` for a 128-beam Ouster lidar).
        - `imu/args/port` and `radar/args/radar/port` should match the serial ports that the IMU and radar mount to, respectively, though these ports should be consistent between systems provided no additional serial devices are plugged in.

## Connect the controller

1. Connect the controller device to the data collection rig's wifi network.

    - Any device with a web browser can be used as the controller. A smart phone is suggested.
    - The router may take a minute or more to boot when powered on.
    - The network name and password should be labeled on the router; the name should also match the name of the data collection rig, which should be on the front of the main rig.

2. Connect the web browser to `{name}.local:5000` where `{name}` is the name of the data collection computer.

    !!! tip

        This should be written on the data collection computer, and should match the name of the data collection rig.

## Collect data

!!! tip

    Do a dry run of the data collection process before unplugging to go collect data!


1. Select the target drive / output location.

    - By default, a `data` option is always included; this saves data to the local disk in the current working directory. External drives will show up as `/data-a`, `/data-b`, etc.
    - External drives should mount automatically.
    - Options are only loaded on page load; if you plug in an external drive after loading the web app, refresh the page.

    !!! tip
    
        If an external drive still does not show up, verify that the drive has mounted automatically in `/media/rover/...` using the display/keyboard/terminal.

2. **(optional)** Provide a trace name.

    - Use the text input to set a name that the recorded trace will be saved as.
    - If left blank, will be named after the current timestamp in `YYYY-MM-DD.HH-MM-SS` format.

3. In the web app, press the `start` button.
    
    !!! info "Expected Behavior"

        - Text should begin to scroll on all four sections of the web app; the frequency and utilization of data collection for each sensor are regularly logged.

    !!! info

        Blue messages indicate normal information.
    
    !!! warning
    
        Yellow messages indicate warnings.

        * The IMU usually gives an `invalid checksum` warning when starting data collection; if this only occurs once, it can be ignored.
        * The radar may give a `missing packets` warning from time to time. If this occurs infrequently (once every few blue status messages, i.e. no more than a few times per minute), it can be ignored. Missing packet warnings are observed to be correlated with certain locations, and perhaps temperature/humidity; the exact cause is not known.
        * If the data collection frequency drops below 99% of the expected frequency, a warning is logged. This can be ignored if intermittent.

    !!! failure "Error"

        Red messages indicate errors. **Upon receiving an error, you should immediately stop data collection until the cause is identified**.

        - If you attempt to start data collection when one is already running, or start a data collection with a name which is already used, you will receive an error.
        - If you attempt to stop data collection when no data collection is running, you will also receive an error.
        - Any other errors are likely system and/or hardware problems. If a full reboot (turn off computer, unplug the battery, and return to step 0.) does not fix the problem, please report this.
        - If the data collection frequency drops below 90% of the expected frequency, an error is logged. If this happens outside of initialization (e.g. not in the first few messages), please report this.

4. Press stop.

    !!! info "Expected Behavior"

        Each section should stop collecting data.
    
    !!! note
    
        If you reload the page while data collection is active, the `STOP` button will be greyed out, since the web app does not save its state. In this case, press `START`; this will cause an error, which can be ignored. The `STOP` button will then be enabled.

## Clean up

1. Ensure that data collection is not active.

2. Plug the monitor/keyboard/mouse back in. Kill the server (with `ctrl+C`), then shut down the computer with `sudo shutdown now`.

3. Unplug the power strip from the battery.

    - Return the battery to its charger.
    - If using a shared controller phone, please return the phone to its charger as well.

4. Replace the protective cover on the radar.

## Troubleshooting

If some data types (especially Lidar) are not being collected as expected:

- Check all cables for loose connections or damaged connectors.
- Reboot the data collection computer.
- Reboot the rig (by turning off the computer, and unpowering all sensors, e.g. by unplugging the root power strip).

For radar-related issues:

- See the [`xwr` troubleshooting documentation](https://wiselabcmu.github.io/xwr/#troubleshooting).

If the camera is not coming through at full rate:

- Check for damaged/kinked cables and loose connections.
- Remove all of the camera-related cables and connectors from the internal cavity, lay them out flat, then test.
