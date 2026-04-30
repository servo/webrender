# The spatial tree

The [spatial tree](https://searchfox.org/mozilla-central/search?path=&q=SpatialTree) contains the hierarchy of transformations that all rendering primitives are associated with. It is built along with the scene and its content is queried during frame building. The values of animated transforms can be modified between frames.

Internally, it defines two concepts :
 - Coordinate **spaces**,
 - Coordinate **systems**.

Each coordinate space is represented by a spatial node. Outside code refer to spatial nodes using spatial node indices, and almost anything that manipulates coordinates in 2D or 3D does so relative to a specific spatial node of the spatial tree.

All coordinates in WebRender are relative to a specific coordinate space. All processing happens in a specific coordinate space. In the code, a [`spatial node index`](https://searchfox.org/mozilla-central/search?path=&q=SpatialNodeIndex) is used to refer to a spatial node. All coordinate spaces are represented in the spatial tree, with one important exception: device space. We'll come back to this later in this post.

In web content, the vast majority of transformations can be represented with 2D scales of offsets, but they can also sometimes be more complex arbitrary 2D or 3D transformations. To model that, the spatial tree represents all coordinate **spaces** as 2D scale/offset transformations relative to a coordinate **system**. A coordinate systems can represent any type of transformation.

Consider the following pseudo-scene:

```
* red rectangle
* stacking context { transform: scale(3.0) }
    * green rectangle
    * stacking context { transform: scale(0.5) }
        * blue rectangle
    * stacking context { transform: rotate(-10) }
		* yellow rectangle
```

The structure of the spatial tree would look like this:

```
systems:
    s0: (root)
        s1: rotate(-10)
spaces:
    #0: system: s0
    #1: system: s0, scale(3.0)
    #2: system: s0, scale(1.5)
    #3: system: s1
```

Note how the original `scale(0.5)` relative to `scale(3.0)` in our scene was flattened into `scale(1.5)` relative to the same coordinate system (`s0`).

WebRender does a lot of transformations from a coordinate space to another there are three cases:
- A pretty common one is when the source and destination spaces are the same. The transformation is a no-op.
- If the coordinate spaces (let's call them `A` and `B`) are different but attached to the same coordinate system, going from `A` to `B` is expressed via a simple scale+offset transformation: It is the concatenation of `A.content_transform.inverse()` and `B.content_transform`, or in other words the transform maps from `A` to its coordinate system and from there to `B`.
- Finally, if the coordinate spaces are in different coordinate systems then a full 4x4 matrix transform is produced and involves walking the tree of coordinate systems from from the source to the destination. This case is a lot less common than the other two.

The spatial tree is defined during scene building, but the value of the transforms can change between frame without having to re-build the scene.
# Coordinate types

Juggling between coordinate spaces can be quite error-prone. To facilitate catching mistakes, WebRender leverages Rust's type system using the tagged types from the [euclid](docs.rs/euclid/latest/euclid/) linear algebra crate. In practice what we have is generic types such as `euclid::Point2D<Scalar, Space>` which we use via [a number of aliases](https://searchfox.org/firefox-main/source/gfx/wr/webrender_api/src/units.rs#1) such as

```rust
pub type LayoutPoint = euclid::Point2D<f32, LayoutPixel>;
pub type DeviceIntPoint = euclid::Point2D<i32, DevicePixel>;
// etc.
```

These tags make sure that we don't, for example, add a `LayoutVector` to a `DeviceVector` by mistake. 

The most Important coordinate types tags are `WorldPixel`, `LayoutPixel`, `DevicePixel`, `RasterPixel` and `PicturePixel`. It is important to understand that there isn't a one to one correspondence between these coordinate tags and nodes of the spatial tree.
Coordinate tags provide an interpretation of the meaning of a spatial node in a specific context. It will be easier to understand after going through each of these tags, but as a word caution, I will be using the overloaded "space" terminology ("Device space", "Layout space", etc.) to refer to these types rather than the coordinate spaces in the spatial tree. There is an unfortunate overlap here, but it is all over WebRender's code, and this post reflects that.

## Layout

Layout space (often also called "local space") is what display items are specified in at WebRender's public API boundary. There isn't one layout space: all coordinates in layout space are always relative to a specific spatial node. In CSS parlance, layout space coordinates are always relative to a stacking context.

I think of layout space as the relative coordinates at the beginning of the chain of transformations.
Note that display items parameters such as gradient endpoints are typically specified using types such as `LayoutPoint`, but they are actually relative to the bounds of the display item instead of being relative to a spatial node like the rest of the code that explicitly uses layout coordinates.

## Device

At the opposite end of the chain of transformation is device space. Device space refers to coordinate spaces that map to actual pixels.

We typically do computation in device space when we need to know how coordinates will land on the pixel grid of a destination texture or, since one unit in device space corresponds to one pixel, to decide what resolution to render some intermediate steps in.

For example consider a complex clip mask that needs to be drawn into an intermediate texture to mask out some content. We could render the mask at any resolution, but in order for the result to not look blurry or pixelated, we want the resolution of the mask to be selected based on the size that it will occupy once applied in the destination texture. Also, if the content is drawn at a fractional offset, the mask should take that offset into account, so device space is a good place to ensure that both the item and its mask align.

Device space special in a few ways, chief among them the fact there isn't a node in the spatial tree that map directly to it.

## Picture

Internally, WebRender refers to surface into which arbitrary content needs to be rendered as ["pictures"](https://searchfox.org/mozilla-central/search?q=PicturePrimitive). Certain things such as filters or complex clips, require content to be drawn into an intermediate picture which is then drawn as an image into its parent picture. Pictures can be arbitrarily nested so they form a tree and all display items are assigned to a picture in that tree.

Like other internal rendering primitive, pictures are associated to a spatial node which defines how to position them. Let's call it the positional node of the picture. We tag coordinates relative to this positional node with the `PicturePixel` tag, they are said to be in "picture space".

In addition, pictures have the concept of a "raster node". The raster node is also a spatial node index, that is used to specify in which coordinate space some important operations are performed. In some cases a picture's positional and raster nodes are the same, sometimes they are different.

## Raster

Raster space is a coordinate space that is used as a reference for rendering multiple items specified in different coordinate spaces into a single surface. Like layout coordinates, raster coordinates are relative to a spatial node. For the purpose of rendering a picture, that particular spatial node is chosen as the reference "raster node" of the picture.

In order to project something into device space, the process is always to first project into raster space, and then multiply by the `device_pixel_scale`.

WebRender's code also mentions "world space". As far as I can tell, raster and world space are the same thing. It looks like the intent for world space was originally to specifically correspond to the raster space of the root of the scene, but it does not hold true in practice. If my understanding is correct, then we should replace all mentions of world space with raster space. In the mean time I consider them to mean the same thing.

Why is raster space different from picture space?

There are some subtle behaviors that WebRender needs to support and for which Firefox has test coverage. Consider the following example:

```
* stacking-context { transform: offset(0.3, 0.0) }
    * stacking-context { filter: opacity(0.5) }
        * blue rectangle (with a 0.3 offset)
```

The second stacking context requires an intermediate picture into which to draw the two rectangles and apply the opacity filter. Both rectangles are pixel-snapped.



The spatial tree looks like:

```
systems:
    s0: (root)

spaces:
    #0: space: s0, (root)
    #1: space: s0, offset(0.3, 0.0)
```

The picture graph:

```
* p0 { positional_node: #0, raster_node: #0 }
    * p1 { positional_node: #1, raster_node: #0 }
```

During scene building, the coordinates of the blue rectangle are adjusted such that once projected into the device space of the root picture, its bounds are snapped to the pixel grid of the root.

If we were to use the filtered surface's spatial node as the raster node, then, since that spatial node has a 0.3 offset, the snapping would not match the pixel grid anymore. An extra offset has to be applied to account for that. That offset is tracked by the snapping transform of the raster node.

We could have used the render task's content offset to compensate, except that for other historical reasons, the content offset can only be whole integers. 
In the future we could change this to be a separate scale+offset per render task that would be used to apply the missing fractional offset from the picture's positional node, relative to the raster root.


Since we have this concept of raster node why do we chose to do some processing in the coordinate space of the picture item instead of, expressing everything directly in raster space?
- I suspect that for a large part it has to do with how likely the child items of a picture are to be using the same spatial node as the picture itself. When computing the size of the picture (which amounts to accumulating all of the item rectangles in the picture), the cheapest coordinate space to do this in is the one that most items use, since transposing coordinates into that space is a no-op.
- It may also be partly historical, since raster space was introduced later to address a specific snapping issue. 

## World

World space refers to the coordinate space of the root of the document. It is typically used when operating in the global space of the document, for example in hit-testing queries or compositing.

## Visibility

Visibility space is the space in which WebRender performs culling, clipping and invalidation.

At the moment this space is always the root (world) space, but the aim is to move to using raster space.

# Miscellaneous

So device space is a little special. Why isn't there a spatial node for it in the spatial tree?

Mainly, I think, because the spatial tree is part of the scene description while device space is evaluated during frame building (when we know the value of each transform which can change between frames).

Although device space sounds like direct texture coordinates where `(0, 0)` would map to one of the corners of the texture, an offset is applied before we finally get the true texture coordinates. This offset is called `content_offset`, it's in device space and is passed to the vertex shader along with the device pixel scale in the render task data. Here are two examples illustrating the importance of the content offset:
- Tiled pictures use a single coordinate space for all tiles, and each tile applies its own offset in the vertex shader via the content offset.
- A lot of render tasks are drawn into texture atlases. We don't create a separate spatial node per render task to position their content correctly relative to the task's position in the atlas. The content offset is used for this purpose.

## How to decide what space to do some computation in

For example, decomposing a primitive into multiple segments can be done in various coordinate spaces.

A few rules of thumb:
 - If you need a coordinate space in which the primitive is axis-aligned, pick layout space.
 - If coordinates need to be in the same space as a pattern, pick layout space.
 - If you need to know the exact projected size in pixels of a primitive, there's a good chance you need to work in device space.
 - If you need to project many sibling items into a coordinate space, picture space has the best chance of not requiring a lot of expensive transforms.
 - If the processing requires a coordinate space global to the entire document (this should be rare), use world space.
 - If the coordinate space must be the one that the shaders manipulate before applying the `device_pixel_scale` and `content_offset`, then pick raster space.

## On the shader side

Most primitives pass coordinates to the vertex shader in layout space. The exception is quad shaders which currently either take layout or device space coordinates. Since patterns are only expressed in layout coordinates, quad shaders take a scale+offset transform that can be applied to the uv coordinates of the patterns. This transform is either identity when the quad coordinates are provided in layout space, or the inverse of the layout-to-device transform when the quad coordinates are provided in device space.


# Pixel snapping

Snapping is a big topic worthy of its own document. However it interacts with coordinate spaces in a somewhat surprising way, so it is worth mentioning here.

In principle it would make sense, for primitives that need, it to be snapped relative to the device space of the parent picture surface, After all snapping is all about aligning with the pixel grid of the texture an item is being rendered into.

However, WebRender currently only snaps relative to the root coordinate space. If a scale+offset transform cannot be established between local and world space, then the primitive is not snapped.
In practice, this means that the general rule for snapping actually only applies to a subset of content (which tends to be the majority of it).
