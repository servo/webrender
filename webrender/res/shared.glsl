/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef WR_FEATURE_TEXTURE_EXTERNAL
// Please check https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
// for this extension.
#extension GL_OES_EGL_image_external_essl3 : require
#endif

// The textureLod() doesn't support samplerExternalOES for WR_FEATURE_TEXTURE_EXTERNAL.
// https://www.khronos.org/registry/OpenGL/extensions/OES/OES_EGL_image_external_essl3.txt
//
// The textureLod() doesn't support sampler2DRect for WR_FEATURE_TEXTURE_RECT, too.
//
// Use texture() instead.
#if defined(WR_FEATURE_TEXTURE_EXTERNAL) || defined(WR_FEATURE_TEXTURE_RECT) || defined(WR_FEATURE_TEXTURE_2D)
#define TEX_SAMPLE(sampler, tex_coord) texture(sampler, tex_coord.xy)
#else
// In normal case, we use textureLod(). We haven't used the lod yet. So, we always pass 0.0 now.
#define TEX_SAMPLE(sampler, tex_coord) textureLod(sampler, tex_coord, 0.0)
#endif

// texelFetchOffset has driver bugs on some Android GPUs (see issue #1694).
// Fallback to texelFetch on mobile GPUs.
#if defined(GL_ES) && GL_ES == 1
    #define TEXEL_FETCH(sampler, position, lod, offset) texelFetch(sampler, position + offset, lod)
#else
    #define TEXEL_FETCH(sampler, position, lod, offset) texelFetchOffset(sampler, position, lod, offset)
#endif

//======================================================================================
// Vertex shader attributes and uniforms
//======================================================================================
#ifdef WR_VERTEX_SHADER
    #define varying out

    // Uniform inputs
    uniform mat4 uTransform;       // Orthographic projection
    uniform float uDevicePixelRatio;

    // Attribute inputs
    in vec3 aPosition;
#endif

//======================================================================================
// Fragment shader attributes and uniforms
//======================================================================================
#ifdef WR_FRAGMENT_SHADER
    precision highp float;

    #define varying in

    // Uniform inputs

    // Fragment shader outputs
    out vec4 oFragColor;
#endif

//======================================================================================
// Shared shader uniforms
//======================================================================================
#if defined(GL_ES)
    #if GL_ES == 1
        #ifdef GL_FRAGMENT_PRECISION_HIGH
        precision highp sampler2DArray;
        #else
        precision mediump sampler2DArray;
        #endif

        // Sampler default precision is lowp on mobile GPUs.
        // This causes RGBA32F texture data to be clamped to 16 bit floats on some GPUs (e.g. Mali-T880).
        // Define highp precision macro to allow lossless FLOAT texture sampling.
        #define HIGHP_SAMPLER_FLOAT highp
    #else
        #define HIGHP_SAMPLER_FLOAT
    #endif
#else
    #define HIGHP_SAMPLER_FLOAT
#endif

#ifdef WR_FEATURE_TEXTURE_2D
uniform sampler2D sColor0;
uniform sampler2D sColor1;
uniform sampler2D sColor2;
#elif defined WR_FEATURE_TEXTURE_RECT
uniform sampler2DRect sColor0;
uniform sampler2DRect sColor1;
uniform sampler2DRect sColor2;
#elif defined WR_FEATURE_TEXTURE_EXTERNAL
uniform samplerExternalOES sColor0;
uniform samplerExternalOES sColor1;
uniform samplerExternalOES sColor2;
#else
uniform sampler2DArray sColor0;
uniform sampler2DArray sColor1;
uniform sampler2DArray sColor2;
#endif

#ifdef WR_FEATURE_DITHERING
uniform sampler2D sDither;
#endif

//======================================================================================
// Interpolator definitions
//======================================================================================

//======================================================================================
// VS only types and UBOs
//======================================================================================

//======================================================================================
// VS only functions
//======================================================================================
