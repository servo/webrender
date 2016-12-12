# webrender
A somewhat incomplete proof of concept GPU renderer for Servo

After updating shaders in webrender, go to servo and:

  * Go to the servo directory and do ./mach update-cargo -p webrender
  * Create a pull request to servo and update the shaders in the servo repository.


# Use Webrender with Servo
To use a custom webrender with servo, go to your servo build directory and:

  * Edit Cargo.toml
  * Add at the end of the file:

```
[replace]
"https://github.com/servo/webrender#0.11.0" = { path = 'Path/To/webrender/webrender/' }
"https://github.com/servo/webrender#webrender_traits:0.11.0" = { path = 'Path/To/webrender/webrender_traits' }
```

  * Build as normal

# Webrender coordinate systems.

The general rule of thumb is that coordinates used in display
lists, clips, viewports, transforms and stacking contexts are always:

 * CSS / Logical pixels.
 * In the local (untransformed) coordinate space of the owning stacking context.
 * Assume that the scroll offset is zero.

The coordinates used in stacking contexts and display lists are logical
units, the same as CSS pixels. They are the same value regardless of the
dpi scaling ratio. The DPI scaling ratio is applied on the GPU as required.

When scrolling occurs, none of the coordinates in the display lists change.
Scrolling is handled internally by tweaking matrices that get sent to the
GPU in order to transform the display items.

There are a small number of APIs (primarily ones that interact with events
such as scroll and mouse clicks etc) that use device pixels (including any
hi-dpi scale factor).
