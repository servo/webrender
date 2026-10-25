/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Contains functionality to help building the render task graph from a series of off-screen
//! surfaces that are created during the prepare pass, and other surface related types and
//! helpers.

use api::units::*;
use crate::command_buffer::{CommandBufferBuilderKind, CommandBufferBuilder, CommandBufferIndex, OpenBackdropCapture};
use crate::internal_types::FastHashMap;
use crate::picture_composite_mode::PictureCompositeMode;
use crate::tile_cache::{TileKey, SubSliceIndex, MAX_COMPOSITOR_SURFACES};
use crate::render_task_graph::{RenderTaskId, RenderTaskGraphBuilder};
use crate::render_task::RenderTaskKind;
use crate::space::SpaceMapper;
use crate::spatial_tree::{CoordinateSpaceMapping, CoordinateSystemId, SpatialTree, SpatialNodeIndex};
use crate::util::{MaxRect, ScaleOffset};
use crate::visibility::{BackdropCaptureIndex, DrawState, PrimitiveDrawHeader, FrameVisibilityContext};
pub use crate::picture_composite_mode::get_surface_rects;

/// The mapping between a raster node's space and the screen framebuffer's device
/// space. That is the root reference frame's space - the root carries no device
/// scale of its own - which is what lets the screen rect be the target of this
/// mapping.
///
/// Only a 2D scale and offset makes a rect mapped through this a sound bound in
/// the other space: `map` and `unmap` take the bounding box of the four mapped
/// corners, which is the exact image for a 2D scale+offset but *not* a superset
/// of it once the mapping rotates or has perspective. The image of an
/// axis-aligned rect is then a general quadrilateral, possibly unbounded, and
/// its corners do not bound it; culling against that would drop content that is
/// on screen. Callers must check `as_2d_scale_offset` before trusting the
/// result for culling.
fn raster_to_root_mapper(
    raster_spatial_node_index: SpatialNodeIndex,
    bounds: DeviceRect,
    spatial_tree: &SpatialTree,
) -> SpaceMapper<RasterPixel, DevicePixel> {
    SpaceMapper::new_with_target(
        spatial_tree.root_reference_frame_index(),
        raster_spatial_node_index,
        bounds,
        spatial_tree,
    )
}

/// The mapping from a surface's picture space into its device space, that is its
/// raster space scaled by `device_pixel_scale`.
///
/// Always a 2D scale+offset, which is what makes a device rect and a picture
/// rect describe the same region rather than one being a looser bound on the
/// other, and what makes the mapping distribute over intersection.
/// `PictureInstance::assign_surface` only lets the raster node differ from the
/// surface node for a root-snapping surface, which requires the surface node to
/// be in the root coordinate system - and a node there relates to the root by a
/// scale+offset by construction. Otherwise the two nodes are the same and the
/// mapping is the device scale alone.
fn picture_to_device_mapping(
    surface_spatial_node_index: SpatialNodeIndex,
    raster_spatial_node_index: SpatialNodeIndex,
    device_pixel_scale: DevicePixelScale,
    spatial_tree: &SpatialTree,
) -> ScaleOffset {
    let picture_to_raster = if raster_spatial_node_index == surface_spatial_node_index {
        ScaleOffset::identity()
    } else {
        debug_assert_eq!(
            device_pixel_scale.0, 1.0,
            "a surface that rasterizes in another node's space carries no device scale",
        );
        debug_assert_eq!(
            raster_spatial_node_index,
            spatial_tree.root_reference_frame_index(),
            "the only raster node a surface does not share is the root",
        );

        match spatial_tree.get_relative_transform(
            surface_spatial_node_index,
            raster_spatial_node_index,
        ) {
            CoordinateSpaceMapping::Local => ScaleOffset::identity(),
            CoordinateSpaceMapping::ScaleOffset(scale_offset) => scale_offset,
            CoordinateSpaceMapping::Transform(..) => {
                debug_assert!(
                    false,
                    "surface at {:?} rasterizing in {:?} is not axis-aligned in it",
                    surface_spatial_node_index,
                    raster_spatial_node_index,
                );
                ScaleOffset::identity()
            }
        }
    };

    picture_to_raster.then_scale(device_pixel_scale.0)
}

/// Maximum blur radius for blur filter
const MAX_BLUR_RADIUS: f32 = 100.;

/// An index into the surface array
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SurfaceIndex(pub usize);

/// Specify whether a surface allows subpixel AA text rendering.
#[derive(Debug, Copy, Clone)]
pub enum SubpixelMode {
    /// This surface allows subpixel AA text
    Allow,
    /// Subpixel AA text cannot be drawn on this surface
    Deny,
    /// Subpixel AA can be drawn on this surface, if not intersecting
    /// with the excluded regions, and inside the allowed rect.
    Conditional {
        allowed_rect: PictureRect,
        prohibited_rect: PictureRect,
    },
}

/// Information about an offscreen surface. For now,
/// it contains information about the size and coordinate
/// system of the surface. In the future, it will contain
/// information about the contents of the surface, which
/// will allow surfaces to be cached / retained between
/// frames and display lists.
pub struct SurfaceInfo {
    /// A local rect defining the size of this surface, in the
    /// coordinate system of the parent surface. This contains
    /// the unclipped bounding rect of child primitives.
    ///
    /// SNAPTODO: This rect is built by mapping per-cluster bounding
    /// rects (and child-surface coverage rects) into this surface's
    /// picture space via `map_local_to_picture`. Even once the source
    /// cluster bound is a true union of per-prim *snapped* local
    /// rects, the resulting `unclipped_local_rect` is not guaranteed
    /// to be snapped: any 2D transform that isn't an axis-aligned,
    /// integer-pixel translation between the cluster/child-surface
    /// spatial node and this surface's spatial node will produce
    /// sub-pixel edges in picture space. Float blur-margin inflation
    /// inside `composite_mode.get_coverage` can also break the snap.
    /// Consumers that need a snapped value will either need to
    /// re-snap in surface space or restrict the snap path to surfaces
    /// where the cross-space mapping preserves grid alignment (see
    /// `SurfaceInfo.allow_snapping`).
    pub unclipped_local_rect: PictureRect,
    /// The local space coverage of child primitives after they are
    /// are clipped to their owning clip-chain.
    pub clipped_local_rect: PictureRect,
    /// The (conservative) valid part of this surface, in this surface's device
    /// space. Used to reduce the size of render target allocation.
    ///
    /// Device space rather than picture space because every consumer intersects
    /// a primitive's footprint against it to size or place a render task, which
    /// is a device-space rect. The mapping between the two spaces is a 2D
    /// scale+offset (see `picture_to_device`) and so distributes over
    /// intersection, which is what lets the intersection happen on this side of
    /// it without changing the result.
    pub clipping_rect: DeviceRect,
    /// The rectangle to use for culling and clipping, in the local space of
    /// `raster_spatial_node_index`. A primitive outside it cannot affect
    /// anything on screen.
    ///
    /// For a root surface this is the visible region of the screen expressed in
    /// that space. For a child surface it is that region as seen through the
    /// chain of composite modes above it, which is *not* just the part of the
    /// screen the surface covers: a blur, a drop shadow or an SVG filter graph
    /// samples outside its own destination, so content that is off-screen (or
    /// outside the parent's culling rect) still contributes through them.
    /// `update_culling_rect` expands the rect by what the composite mode reads.
    ///
    /// Never empty as a way of saying "nothing is visible": an empty culling
    /// rect culls the whole surface, so any projection that cannot be computed
    /// falls back to `max_rect` (cull nothing) instead.
    pub culling_rect: RasterRect,
    /// Whether `culling_rect` is the `max_rect` fallback rather than a real
    /// projection of the screen. Instrumentation only: a `max_rect` culling rect
    /// is also legitimate for a surface handed an unbounded screen rect.
    pub culling_rect_projection_failed: bool,
    /// Helper structs for mapping local rects in different
    /// coordinate systems into the picture coordinates.
    pub map_local_to_picture: SpaceMapper<LayoutPixel, PicturePixel>,
    /// This surface's picture space to its device space. Fixed for the frame
    /// once the surface is assigned, except that `get_surface_rects` recomputes
    /// it when it has to scale a surface down to fit `max_surface_size`.
    /// Prefer `map_to_device_rect` over using it directly.
    pub picture_to_device: ScaleOffset,
    /// The positioning node for the surface itself,
    pub surface_spatial_node_index: SpatialNodeIndex,
    /// The rasterization root for this surface.
    pub raster_spatial_node_index: SpatialNodeIndex,
    /// The device pixel ratio specific to this surface.
    pub device_pixel_scale: DevicePixelScale,
    /// The scale factors of the surface to world transform. Child surfaces
    /// multiply their own child-to-parent scale by this to obtain
    /// child-to-device, so it must describe this surface's space all the way to
    /// device space.
    pub world_scale_factors: (f32, f32),
    /// The remaining per-axis scale from the space this surface rasterizes in to
    /// device space, i.e. `local_scale * blur_scale_factors` is the surface's
    /// full local-to-device scale. Only blur radius clamping uses it, and it is
    /// kept separate from `world_scale_factors` because a root-snapping surface
    /// rasterizes in root space (so this is one) while still needing to report a
    /// real scale to its children.
    pub blur_scale_factors: (f32, f32),
    /// Local scale factors surface to raster transform
    pub local_scale: (f32, f32),
    /// If true, we know this surface is completely opaque.
    pub is_opaque: bool,
    /// Whether content rasterized into this surface is snapped to the device
    /// pixel grid at frame time. True for tile caches (snapped against the
    /// scroll-stable cache node) and root-snapping surfaces (raster node is
    /// root). False for a non-snapping raster root (preserve-3d / perspective /
    /// huge-scale, `enable_snapping == false`): snapping against its own scaled
    /// node would use only the tiny local scale and collapse content to zero,
    /// so content is left unsnapped there instead.
    pub allow_snapping: bool,
    /// If true, the scissor rect must be set when drawing this surface
    pub force_scissor_rect: bool,
    /// For an SVGFEGraph surface, the mapping from the space the filter
    /// subregions are authored in (the filtered element's spatial node) to this
    /// surface's spatial node. Non-identity for backdrop filters, whose graph
    /// composites in backdrop-root space; it is a full scale+offset because an
    /// intervening reference frame may scale (e.g. pdf.js scales its text
    /// spans), so a translation alone is not enough. All SVGFE coverage paths
    /// map the subregions through this so they line up with the geometry.
    pub svgfe_source_map: ScaleOffset,
    /// Whether this is the surface of a picture in a backdrop-filter chain that
    /// rasterizes like the surface it reads its backdrop from.
    pub in_backdrop_chain: bool,
}

impl SurfaceInfo {
    pub fn new(
        surface_spatial_node_index: SpatialNodeIndex,
        raster_spatial_node_index: SpatialNodeIndex,
        global_culling_rect: DeviceRect,
        spatial_tree: &SpatialTree,
        device_pixel_scale: DevicePixelScale,
        world_scale_factors: (f32, f32),
        blur_scale_factors: (f32, f32),
        local_scale: (f32, f32),
        allow_snapping: bool,
        force_scissor_rect: bool,
    ) -> Self {
        let map_surface_to_root = SpaceMapper::new_with_target(
            spatial_tree.root_reference_frame_index(),
            surface_spatial_node_index,
            global_culling_rect,
            spatial_tree,
        );

        let pic_bounds = map_surface_to_root
            .unmap(&map_surface_to_root.bounds)
            .unwrap_or_else(PictureRect::max_rect);

        let map_local_to_picture = SpaceMapper::new(
            surface_spatial_node_index,
            pic_bounds,
        );

        // The culling rect is the screen, expressed in raster space.
        let map_raster_to_root = raster_to_root_mapper(
            raster_spatial_node_index,
            global_culling_rect,
            spatial_tree,
        );

        // A raster node in the root coordinate system always gives a
        // scale+offset, so the guard only bites for a raster root established
        // inside a 3D context - where the answer is to cull nothing.
        let raster_to_root = map_raster_to_root.as_2d_scale_offset();
        let projected = raster_to_root
            .and_then(|_| map_raster_to_root.unmap(&global_culling_rect));

        let mut culling_rect_projection_failed = false;
        let culling_rect = match projected {
            Some(rect) => rect,
            None => {
                culling_rect_projection_failed = true;
                // Cull nothing rather than everything; see `culling_rect`.
                debug_assert_ne!(
                    spatial_tree
                        .get_spatial_node(raster_spatial_node_index)
                        .coordinate_system_id,
                    CoordinateSystemId::root(),
                    "raster node in the root coordinate system must give an exact culling rect",
                );
                RasterRect::max_rect()
            }
        };

        // The culling rect has to describe the same region as the screen rect it
        // was derived from, so mapping it back must still cover the screen. A vis
        // space that lost part of the screen on the way in would cull content
        // that is genuinely visible - the failure mode that matters when the vis
        // node moves away from the root.
        //
        // Only checkable for a rect that came from a real projection. The
        // `max_rect` fallback culls nothing by construction, and projecting it
        // forward says nothing either way: `project_rect` clips against the near
        // plane, so a near-plane-crossing transform maps `max_rect` to a bounded
        // rect that need not cover the screen.
        #[cfg(debug_assertions)]
        if let Some(round_trip) = Some(&culling_rect)
            .filter(|_| !culling_rect_projection_failed)
            .and_then(|rect| map_raster_to_root.map(rect))
        {
            // The round trip is only exact while the raster space stays near the
            // origin. `unmap` divides the screen by the mapping's transform and
            // `map` multiplies it back, each rounding at the magnitude of the
            // translation involved, so a surface at a large scroll or pinch-zoom
            // offset loses a fraction of a pixel even when its space is a sound
            // pre-image of the screen. Widen the tolerance with that magnitude so
            // f32 rounding is not mistaken for culled content.
            const EPSILON: f32 = 0.05;
            let epsilon = EPSILON + raster_to_root.map_or(0.0, |scale_offset| {
                scale_offset.offset.x.abs()
                    .max(scale_offset.offset.y.abs())
                    .max(scale_offset.scale.x.abs())
                    .max(scale_offset.scale.y.abs())
                    * EPSILON * 4.0
            });
            debug_assert!(
                round_trip.inflate(epsilon, epsilon).contains_box(&global_culling_rect),
                "vis culling rect {:?} loses part of the screen {:?} (round trip {:?})",
                culling_rect,
                global_culling_rect,
                round_trip,
            );
        }

        SurfaceInfo {
            unclipped_local_rect: PictureRect::zero(),
            clipped_local_rect: PictureRect::zero(),
            is_opaque: false,
            clipping_rect: DeviceRect::zero(),
            map_local_to_picture,
            picture_to_device: picture_to_device_mapping(
                surface_spatial_node_index,
                raster_spatial_node_index,
                device_pixel_scale,
                spatial_tree,
            ),
            raster_spatial_node_index,
            surface_spatial_node_index,
            device_pixel_scale,
            world_scale_factors,
            blur_scale_factors,
            local_scale,
            allow_snapping,
            force_scissor_rect,
            svgfe_source_map: ScaleOffset::identity(),
            in_backdrop_chain: false,
            culling_rect,
            culling_rect_projection_failed,
        }
    }

    /// Clamps the blur radius depending on scale factors.
    pub fn clamp_blur_radius(
        &self,
        x_blur_radius: f32,
        y_blur_radius: f32,
    ) -> (f32, f32) {
        // Clamping must occur after scale factors are applied, but scale factors are not applied
        // until later on. To clamp the blur radius, we first apply the scale factors and then clamp
        // and finally revert the scale factors.

        let sx_blur_radius = x_blur_radius * self.local_scale.0;
        let sy_blur_radius = y_blur_radius * self.local_scale.1;

        let largest_scaled_blur_radius = f32::max(
            sx_blur_radius * self.blur_scale_factors.0,
            sy_blur_radius * self.blur_scale_factors.1,
        );

        if largest_scaled_blur_radius > MAX_BLUR_RADIUS {
            let sf = MAX_BLUR_RADIUS / largest_scaled_blur_radius;
            (x_blur_radius * sf, y_blur_radius * sf)
        } else {
            // Return the original blur radius to avoid any rounding errors
            (x_blur_radius, y_blur_radius)
        }
    }

    /// Derive this surface's culling rect from the one the parent surface uses.
    ///
    /// The parent's rect is in the parent's raster space, which is not this
    /// surface's whenever this surface establishes its own raster root, so it is
    /// mapped across before anything else looks at it.
    pub fn update_culling_rect(
        &mut self,
        parent_raster_spatial_node_index: SpatialNodeIndex,
        parent_culling_rect: RasterRect,
        composite_mode: &PictureCompositeMode,
        frame_context: &FrameVisibilityContext,
    ) {
        // A parent that culls nothing gives a child that culls nothing. Taking
        // the general path instead would round-trip `max_rect` through
        // projections that clip against the near plane, and the result need not
        // still cover everything.
        if parent_culling_rect == RasterRect::max_rect() {
            self.culling_rect = parent_culling_rect;
            return;
        }

        let parent_culling_rect = if parent_raster_spatial_node_index == self.raster_spatial_node_index {
            parent_culling_rect
        } else {
            // Cross between the two raster spaces via the screen. The spatial
            // tree only relates a node to one of its ancestors, and neither
            // raster node need be an ancestor of the other, but both always
            // relate to the root.
            let map_parent_to_root = raster_to_root_mapper(
                parent_raster_spatial_node_index,
                frame_context.global_screen_device_rect,
                frame_context.spatial_tree,
            );
            let map_raster_to_root = raster_to_root_mapper(
                self.raster_spatial_node_index,
                frame_context.global_screen_device_rect,
                frame_context.spatial_tree,
            );

            let projected = map_parent_to_root
                .as_2d_scale_offset()
                .and_then(|_| map_parent_to_root.map(&parent_culling_rect))
                .and_then(|device_rect| {
                    map_raster_to_root
                        .as_2d_scale_offset()
                        .and_then(|_| map_raster_to_root.unmap(&device_rect))
                });

            match projected {
                Some(rect) => rect,
                None => {
                    // Cull nothing rather than everything; see `culling_rect`.
                    self.culling_rect = RasterRect::max_rect();
                    return;
                }
            }
        };

        // Content outside the region this surface contributes to can still be
        // sampled by it: a blur, a drop shadow or an SVG filter graph reads
        // outside its own destination. Expand by what the composite mode reads,
        // in surface space where those amounts are expressed, so the content
        // feeding those samples stays inside the culling rect.
        let map_surface_to_raster: SpaceMapper<PicturePixel, RasterPixel> = SpaceMapper::new_with_target(
            self.raster_spatial_node_index,
            self.surface_spatial_node_index,
            parent_culling_rect,
            frame_context.spatial_tree,
        );

        // Unmapping to surface space may be quite conservative in the case of a
        // complex transform, especially perspective.
        let expanded = map_surface_to_raster
            .unmap(&parent_culling_rect)
            .map(|local_rect| composite_mode.get_required_source_rect(self, local_rect.cast_unit()))
            .and_then(|required_rect| map_surface_to_raster.map(&required_rect.cast_unit()));

        // A failed mapping must not leave the un-expanded rect in place: that is
        // exactly the rect that culls the content the expansion exists to keep.
        self.culling_rect = expanded.unwrap_or_else(RasterRect::max_rect);
    }

    /// Re-derive `picture_to_device` after a change to `device_pixel_scale` or
    /// `raster_spatial_node_index`.
    pub fn update_picture_to_device_mapping(
        &mut self,
        spatial_tree: &SpatialTree,
    ) {
        self.picture_to_device = picture_to_device_mapping(
            self.surface_spatial_node_index,
            self.raster_spatial_node_index,
            self.device_pixel_scale,
            spatial_tree,
        );
    }

    /// Map a rect in this surface's picture space into its device space.
    pub fn map_to_device_rect(
        &self,
        picture_rect: &PictureRect,
    ) -> DeviceRect {
        self.picture_to_device.map_rect(picture_rect)
    }

    /// Map a rect in this surface's device space back into its picture space.
    pub fn device_to_picture_rect(
        &self,
        device_rect: &DeviceRect,
    ) -> PictureRect {
        // Content on a surface with no device scale should have been culled out
        // earlier: there is no device space to come back from.
        assert!(self.device_pixel_scale.0 > 0.0);

        self.picture_to_device.unmap_rect(device_rect)
    }

    /// `clipping_rect` in this surface's picture space, for the few consumers
    /// that need to relate it to another surface's picture space rather than
    /// intersect it with a device rect.
    pub fn clipping_rect_in_picture_space(&self) -> PictureRect {
        // `max_rect` means "clip nothing", which has to survive as `max_rect`
        // rather than be divided by the device scale.
        if self.clipping_rect == DeviceRect::max_rect() {
            return PictureRect::max_rect();
        }

        self.device_to_picture_rect(&self.clipping_rect)
    }

    /// Clip a device rect and round it out to a device rect suitable for
    /// allocating a child off-screen surface of this surface (e.g. for
    /// clip-masks)
    pub fn get_surface_rect(
        &self,
        local_rect: &PictureRect,
    ) -> Option<DeviceIntRect> {
        // The content should have been culled out earlier.
        assert!(self.device_pixel_scale.0 > 0.0);

        let device_rect = self
            .map_to_device_rect(local_rect)
            .intersection(&self.clipping_rect)?;

        let surface_rect = device_rect.round_out().to_i32();
        if surface_rect.is_empty() {
            // `local_rect` may have non-empty size that is very close to zero.
            // Due to limited arithmetic precision, the mapping might transform
            // the near-zero-sized rect into a zero-sized one.
            return None;
        }

        Some(surface_rect)
    }
}

// Information about the render task(s) for a given tile
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SurfaceTileDescriptor {
    /// Target render task for commands added to this tile.
    pub current_task_id: RenderTaskId,
    /// Dirty rect for this tile
    pub dirty_rect: PictureRect,
}

/// A task besides its tiles that a tiled surface's primitives are drawn into.
#[cfg_attr(feature = "capture", derive(Serialize))]
#[cfg_attr(feature = "replay", derive(Deserialize))]
pub struct SurfaceExtraTarget {
    /// The rect of the surface the task covers.
    pub rect: PictureRect,
    pub task_id: RenderTaskId,
}

// Details of how a surface is rendered
pub enum SurfaceDescriptorKind {
    // Picture cache tiles
    Tiled {
        tiles: FastHashMap<TileKey, SurfaceTileDescriptor>,
        extra_targets: Vec<SurfaceExtraTarget>,
    },
    // A single surface (e.g. for an opacity filter)
    Simple {
        render_task_id: RenderTaskId,
        dirty_rect: PictureRect,
    },
    // A surface with 1+ intermediate tasks (e.g. blur)
    Chained {
        render_task_id: RenderTaskId,
        root_task_id: RenderTaskId,
        dirty_rect: PictureRect,
    },
}

// Describes how a surface is rendered
pub struct SurfaceDescriptor {
    kind: SurfaceDescriptorKind,
}

impl SurfaceDescriptor {
    /// The task this surface's primitives are drawn into.
    pub fn content_task_id(&self) -> RenderTaskId {
        match self.kind {
            SurfaceDescriptorKind::Tiled { .. } => panic!("bug: tiled surfaces have no single content task"),
            SurfaceDescriptorKind::Simple { render_task_id, .. } |
            SurfaceDescriptorKind::Chained { render_task_id, .. } => render_task_id,
        }
    }

    // Create a picture cache tiled surface
    pub fn new_tiled(
        tiles: FastHashMap<TileKey, SurfaceTileDescriptor>,
        extra_targets: Vec<SurfaceExtraTarget>,
    ) -> Self {
        SurfaceDescriptor {
            kind: SurfaceDescriptorKind::Tiled {
                tiles,
                extra_targets,
            },
        }
    }

    // Create a chained surface (e.g. blur)
    pub fn new_chained(
        render_task_id: RenderTaskId,
        root_task_id: RenderTaskId,
        dirty_rect: PictureRect,
    ) -> Self {
        SurfaceDescriptor {
            kind: SurfaceDescriptorKind::Chained {
                render_task_id,
                root_task_id,
                dirty_rect,
            },
        }
    }

    // Create a simple surface (e.g. opacity)
    pub fn new_simple(
        render_task_id: RenderTaskId,
        dirty_rect: PictureRect,
    ) -> Self {
        SurfaceDescriptor {
            kind: SurfaceDescriptorKind::Simple {
                render_task_id,
                dirty_rect,
            },
        }
    }
}

// Describes a list of command buffers that we are adding primitives to
// for a given surface. These are created from a command buffer builder
// as an optimization - skipping the indirection pic_task -> cmd_buffer_index
struct CommandBufferTargets {
    available_cmd_buffers: Vec<Vec<(PictureRect, CommandBufferIndex)>>,
}

impl CommandBufferTargets {
    fn new() -> Self {
        CommandBufferTargets {
            available_cmd_buffers: vec![Vec::new(); MAX_COMPOSITOR_SURFACES+1],
        }
    }

    fn init(
        &mut self,
        cb: &CommandBufferBuilder,
        rg_builder: &RenderTaskGraphBuilder,
    ) {
        for available_cmd_buffers in &mut self.available_cmd_buffers {
            available_cmd_buffers.clear();
        }

        match cb.kind {
            CommandBufferBuilderKind::Tiled { ref tiles, ref extra_targets } => {
                for (key, desc) in tiles {
                    let task = rg_builder.get_task(desc.current_task_id);
                    match task.kind {
                        RenderTaskKind::Picture(ref info) => {
                            let available_cmd_buffers = &mut self.available_cmd_buffers[key.sub_slice_index.as_usize()];
                            available_cmd_buffers.push((desc.dirty_rect, info.cmd_buffer_index));
                        }
                        _ => unreachable!("bug: not a picture"),
                    }
                }
                for target in extra_targets {
                    match rg_builder.get_task(target.task_id).kind {
                        RenderTaskKind::Picture(ref info) => {
                            for available_cmd_buffers in &mut self.available_cmd_buffers {
                                available_cmd_buffers.push((target.rect, info.cmd_buffer_index));
                            }
                        }
                        _ => unreachable!("bug: not a picture"),
                    }
                }
            }
            CommandBufferBuilderKind::Simple { render_task_id, dirty_rect, .. } => {
                let task = rg_builder.get_task(render_task_id);
                match task.kind {
                    RenderTaskKind::Picture(ref info) => {
                        for sub_slice_buffer in &mut self.available_cmd_buffers {
                            sub_slice_buffer.push((dirty_rect, info.cmd_buffer_index));
                        }
                    }
                    _ => unreachable!("bug: not a picture"),
                }
            }
            CommandBufferBuilderKind::Invalid => {}
        };
    }

    /// For a given rect and sub-slice, get a list of command buffers to write commands to
    fn get_cmd_buffer_targets_for_rect(
        &mut self,
        rect: &PictureRect,
        sub_slice_index: SubSliceIndex,
        targets: &mut Vec<CommandBufferIndex>,
    ) -> bool {

        for (dirty_rect, cmd_buffer_index) in &self.available_cmd_buffers[sub_slice_index.as_usize()] {
            if dirty_rect.intersects(rect) {
                targets.push(*cmd_buffer_index);
            }
        }

        !targets.is_empty()
    }
}

// Main helper interface to build a graph of surfaces.
pub struct SurfaceBuilder {
    // The currently set cmd buffer targets (updated during push/pop)
    current_cmd_buffers: CommandBufferTargets,
    // Stack of surfaces that are parents to the current targets
    builder_stack: Vec<CommandBufferBuilder>,
}

impl SurfaceBuilder {
    pub fn new() -> Self {
        SurfaceBuilder {
            current_cmd_buffers: CommandBufferTargets::new(),
            builder_stack: Vec::new(),
        }
    }

    /// Open a backdrop-filter capture on the current surface. Until it is
    /// closed, each primitive prepared on the surface that is behind the
    /// element is drawn into it too.
    pub fn open_backdrop_capture(&mut self, capture: OpenBackdropCapture) {
        self.builder_stack.last_mut().unwrap().backdrop_captures.push(capture);
    }

    /// Close the capture at `index`, open on the current surface or one below
    /// it, and return it.
    pub fn close_backdrop_capture(&mut self, index: BackdropCaptureIndex) -> Option<OpenBackdropCapture> {
        for builder in self.builder_stack.iter_mut().rev() {
            if let Some(position) = builder.backdrop_captures.iter().position(|capture| capture.index == index) {
                return Some(builder.backdrop_captures.swap_remove(position));
            }
        }
        None
    }

    /// Make the open captures the primitive being prepared is drawn into
    /// depend on `task_id`, as the current surface does.
    fn add_backdrop_capture_dependency(&mut self, task_id: RenderTaskId) {
        if let Some(builder) = self.builder_stack.last_mut() {
            for capture in builder.backdrop_captures.iter_mut().filter(|capture| capture.drawn_into) {
                capture.dependencies.push(task_id);
            }
        }
    }

    pub fn push_surface(
        &mut self,
        surface_index: SurfaceIndex,
        clipping_rect: DeviceRect,
        descriptor: Option<SurfaceDescriptor>,
        surfaces: &mut [SurfaceInfo],
        rg_builder: &RenderTaskGraphBuilder,
    ) {
        // Init the surface
        surfaces[surface_index.0].clipping_rect = clipping_rect;

        let builder = if let Some(descriptor) = descriptor {
            match descriptor.kind {
                SurfaceDescriptorKind::Tiled { tiles, extra_targets } => {
                    CommandBufferBuilder::new_tiled(
                        tiles,
                        extra_targets,
                    )
                }
                SurfaceDescriptorKind::Simple { render_task_id, dirty_rect, .. } => {
                    CommandBufferBuilder::new_simple(
                        render_task_id,
                        None,
                        dirty_rect,
                    )
                }
                SurfaceDescriptorKind::Chained { render_task_id, root_task_id, dirty_rect, .. } => {
                    CommandBufferBuilder::new_simple(
                        render_task_id,
                        Some(root_task_id),
                        dirty_rect,
                    )
                }
            }
        } else {
            CommandBufferBuilder::empty()
        };

        self.current_cmd_buffers.init(&builder, rg_builder);
        self.builder_stack.push(builder);
    }

    // Add a child render task (e.g. a render task cache item, or a clip mask) as a
    // dependency of the current surface
    pub fn add_child_render_task(
        &mut self,
        child_task_id: RenderTaskId,
        rg_builder: &mut RenderTaskGraphBuilder,
    ) {
        self.add_backdrop_capture_dependency(child_task_id);

        self.builder_stack.last().unwrap().kind.for_each_task(|task_id| {
            rg_builder.add_dependency(task_id, child_task_id);
        });
    }

    // Add a picture render task as a dependency of the task this surface's
    // primitives are drawn into, once the surface is popped. It's currently only
    // needed for drop-shadow effects.
    pub fn add_picture_render_task(
        &mut self,
        child_task_id: RenderTaskId,
    ) {
        self.add_backdrop_capture_dependency(child_task_id);
        self.builder_stack
            .last_mut()
            .unwrap()
            .extra_dependencies
            .push(child_task_id);
    }

    // Get a list of command buffer indices that primitives should be pushed
    // to for a given current visbility / dirty state
    pub fn get_cmd_buffer_targets_for_prim(
        &mut self,
        vis: &PrimitiveDrawHeader,
        targets: &mut Vec<CommandBufferIndex>,
    ) -> bool {
        targets.clear();

        let has_targets = match vis.state {
            DrawState::Unset => {
                panic!("bug: invalid vis state");
            }
            DrawState::Culled => {
                false
            }
            DrawState::Visible { sub_slice_index, .. } => {
                self.current_cmd_buffers.get_cmd_buffer_targets_for_rect(
                    &vis.clip_chain.pic_coverage_rect,
                    sub_slice_index,
                    targets,
                )
            }
            DrawState::PassThrough => {
                true
            }
        };

        // A primitive drawn on this surface is drawn into the open captures it
        // is behind too.
        if let Some(builder) = self.builder_stack.last_mut() {
            let is_drawn = !targets.is_empty();
            for capture in &mut builder.backdrop_captures {
                capture.drawn_into = is_drawn &&
                    vis.prim_instance_index.0 < capture.capture_instance_index.0 &&
                    capture.region.intersects(&vis.clip_chain.pic_coverage_rect);
                if capture.drawn_into {
                    targets.push(capture.cmd_buffer_index);
                }
            }
        }

        has_targets
    }

    /// Draw what is prepared next into the capture at `index`, open on the
    /// current surface, and none of its other open captures, until the next
    /// primitive or `None`.
    pub fn select_backdrop_capture(&mut self, index: Option<BackdropCaptureIndex>) {
        for capture in &mut self.builder_stack.last_mut().unwrap().backdrop_captures {
            capture.drawn_into = Some(capture.index) == index;
        }
    }

    pub fn pop_empty_surface(&mut self) {
        self.builder_stack.pop().unwrap();
    }

    // Finish adding primitives and child tasks to a surface and pop it off the stack
    pub fn pop_surface(
        &mut self,
        rg_builder: &mut RenderTaskGraphBuilder,
    ) {
        let builder = self.builder_stack.pop().unwrap();

        if let CommandBufferBuilderKind::Simple { render_task_id, root_task_id, .. } = builder.kind {
            let child_task_id = root_task_id.unwrap_or(render_task_id);
            self.add_backdrop_capture_dependency(child_task_id);
            self.builder_stack.last().unwrap().kind.for_each_task(|parent_task_id| {
                rg_builder.add_dependency(parent_task_id, child_task_id);
            });
        }

        // Step through the dependencies for this builder and add them to the finalized
        // render task root(s) for this surface
        builder.kind.for_each_task(|surface_task_id| {
            for task_id in &builder.extra_dependencies {
                rg_builder.add_dependency(surface_task_id, *task_id);
            }
        });

        // Set up the cmd-buffer targets to write prims into the popped surface
        self.current_cmd_buffers.init(
            self.builder_stack.last().unwrap_or(&CommandBufferBuilder::empty()), rg_builder
        );
    }

    pub fn finalize(self) {
        assert!(self.builder_stack.is_empty());
    }
}


pub fn calculate_screen_uv(
    p: DevicePoint,
    clipped: DeviceRect,
) -> DeviceHomogeneousVector {
    // TODO(gw): Switch to a simple mix, no bilerp / homogeneous vec needed anymore
    DeviceHomogeneousVector::new(
        (p.x - clipped.min.x) / (clipped.max.x - clipped.min.x),
        (p.y - clipped.min.y) / (clipped.max.y - clipped.min.y),
        0.0,
        1.0,
    )
}
