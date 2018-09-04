/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include shared,prim_shared

varying vec3 vUv;
flat varying vec2 vSrcSizeInv;
flat varying vec4 vSrcRect;
// The coefficient and `exp(2.0 * coefficient)`, in that order.
flat varying vec2 vCoefficients;
flat varying int vVertical;

#ifdef WR_FEATURE_COLOR_TARGET
#define TEXTURE_SIZE()  vec2(textureSize(sCacheRGBA8, 0).xy)
#else
#define TEXTURE_SIZE()  vec2(textureSize(sCacheA8, 0).xy)
#endif

#ifdef WR_VERTEX_SHADER

in int aBlurRenderTaskAddress;
in int aBlurSourceTaskAddress;

struct BlurTask {
    RenderTaskCommonData common_data;
    vec2 coefficients;
    int direction;
};

BlurTask fetchBlurTask(int address) {
    RenderTaskData task_data = fetch_render_task_data(address);
    BlurTask task = BlurTask(task_data.common_data,
                             task_data.data1.xy,
                             int(task_data.data1.z));
    return task;
}

void main(void) {
    BlurTask blurTask = fetchBlurTask(aBlurRenderTaskAddress);
    RenderTaskCommonData srcTask = fetch_render_task_common_data(aBlurSourceTaskAddress);

    RectWithSize srcRect = srcTask.task_rect;
    RectWithSize targetRect = blurTask.common_data.task_rect;

    vec2 position = targetRect.p0 + targetRect.size * aPosition.xy;

    vec4 uvBounds = vec4(srcRect.p0, srcRect.p0 + srcRect.size);

    vUv = vec3(mix(uvBounds.xy, uvBounds.zw, aPosition.xy), srcTask.texture_layer_index);
    vSrcSizeInv = 1.0 / TEXTURE_SIZE();
    vSrcRect = vec4(srcRect.p0, srcRect.p0 + srcRect.size) + vec4(0.5, 0.5, -0.5, -0.5);
    vCoefficients = blurTask.coefficients;
    vVertical = blurTask.direction;

    gl_Position = uTransform * vec4(position, 0.0, 1.0);
}

#endif

#ifdef WR_FRAGMENT_SHADER

#define SUPPORT     4

#ifdef WR_FEATURE_COLOR_TARGET
#define SAMPLE_TYPE vec4
#define SAMPLE_TEXTURE(uv)  texture(sCacheRGBA8, uv)
#else
#define SAMPLE_TYPE float
#define SAMPLE_TEXTURE(uv)  texture(sCacheA8, uv).r
#endif

// Accumulates two texels into the blurred fragment we're building up.
void accumulate(float offset,
                float crossAxisCoord,
                inout vec2 gaussCoefficient,
                inout SAMPLE_TYPE colorSum,
                inout float factorSum) {
    float factorA = gaussCoefficient.x;
    gaussCoefficient *= vec2(gaussCoefficient.y, vCoefficients.y);
    float factorB = gaussCoefficient.x;
    gaussCoefficient *= vec2(gaussCoefficient.y, vCoefficients.y);

    // Compute the texture coordinate that provides the correct linear combination of the two
    // texels in question.
    float factors = factorA + factorB;
    float sampleOffset = offset + factorB / factors;

    vec2 texCoord = vec2(sampleOffset, crossAxisCoord);
    texCoord = clamp(vVertical != 0 ? texCoord.yx : texCoord.xy, vSrcRect.xy, vSrcRect.zw);

    colorSum += factors * SAMPLE_TEXTURE(vec3(texCoord * vSrcSizeInv, vUv.z));
    factorSum += factors;
}

void main(void) {
    // FIXME(pcwalton): We shouldn't end up with zero blur radii in the first place!
    if (vCoefficients.x == 0.0) {
        vec2 texCoord = clamp(vUv.xy, vSrcRect.xy, vSrcRect.zw);
        oFragColor = vec4(SAMPLE_TEXTURE(vec3(texCoord * vSrcSizeInv, vUv.z)));
        return;
    }

    bool vertical = vVertical != 0;
    vec2 axisCoord = vertical ? vUv.yx : vUv.xy;
    float start = floor(axisCoord.x - float(SUPPORT)) + 0.5;

    float offset = start - axisCoord.x;

    // See K. Turkowski, "Incremental Computation of the Gaussian", GPU Gems 3, chapter 40:
    //
    // https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch40.html
    vec2 gaussCoefficient = exp(vCoefficients.x * vec2(offset * offset, 2.0 * offset + 1.0));

    SAMPLE_TYPE colorSum = SAMPLE_TYPE(0.0);
    float factorSum = 0.0;

    for (int i = 0; i < SUPPORT + 1; i++)
        accumulate(start + float(i) * 2.0, axisCoord.y, gaussCoefficient, colorSum, factorSum);

    oFragColor = vec4(colorSum / factorSum);
}

#endif
