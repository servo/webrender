# Reviewing WebRender patches

## Render task sizes

WebRender must never request a render task with an empty size (a width or
height of zero or less). `RenderTask::new_dynamic` asserts on this, and the
render task graph panics if it has to allocate an empty surface. Both are
crashes in release builds.

If the patch adds or changes code that creates render tasks (for example calls
to `RenderTask::new_dynamic`, `RenderTask::new_image`,
`RenderTaskCache::request_render_task`, or other ways of adding a task to the
`RenderTaskGraphBuilder`), check how the task size is computed:

- Find every input to the size: primitive or clip rects, intersections with
  the clip or culling rect, scale factors, rounding, blur radii, offsets.
- Check that no combination of inputs can produce an empty size. Intersections
  that can be empty, rounding down, and scaling by small or zero factors are
  common sources.
- When an empty size is possible, the code must handle it before the request,
  usually by skipping the primitive or the effect, or by clamping the size to
  at least one pixel when the output must still exist.
- Flag any change that removes or weakens an existing empty-size check on a
  path that leads to a render task request.
