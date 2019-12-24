/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#![allow(dead_code)]

use api::ColorF;

// A subset of the standard CSS colors, useful for defining GPU tag colors etc.

pub const INDIGO: ColorF = ColorF {
    r: 0.294117647059,
    g: 0.0,
    b: 0.509803921569,
    a: 1.0,
};
pub const GOLD: ColorF = ColorF {
    r: 1.0,
    g: 0.843137254902,
    b: 0.0,
    a: 1.0,
};
pub const FIREBRICK: ColorF = ColorF {
    r: 0.698039215686,
    g: 0.133333333333,
    b: 0.133333333333,
    a: 1.0,
};
pub const INDIANRED: ColorF = ColorF {
    r: 0.803921568627,
    g: 0.360784313725,
    b: 0.360784313725,
    a: 1.0,
};
pub const YELLOW: ColorF = ColorF {
    r: 1.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};
pub const DARKOLIVEGREEN: ColorF = ColorF {
    r: 0.333333333333,
    g: 0.419607843137,
    b: 0.18431372549,
    a: 1.0,
};
pub const DARKSEAGREEN: ColorF = ColorF {
    r: 0.560784313725,
    g: 0.737254901961,
    b: 0.560784313725,
    a: 1.0,
};
pub const SLATEGREY: ColorF = ColorF {
    r: 0.439215686275,
    g: 0.501960784314,
    b: 0.564705882353,
    a: 1.0,
};
pub const DARKSLATEGREY: ColorF = ColorF {
    r: 0.18431372549,
    g: 0.309803921569,
    b: 0.309803921569,
    a: 1.0,
};
pub const MEDIUMVIOLETRED: ColorF = ColorF {
    r: 0.780392156863,
    g: 0.0823529411765,
    b: 0.521568627451,
    a: 1.0,
};
pub const MEDIUMORCHID: ColorF = ColorF {
    r: 0.729411764706,
    g: 0.333333333333,
    b: 0.827450980392,
    a: 1.0,
};
pub const CHARTREUSE: ColorF = ColorF {
    r: 0.498039215686,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};
pub const MEDIUMSLATEBLUE: ColorF = ColorF {
    r: 0.482352941176,
    g: 0.407843137255,
    b: 0.933333333333,
    a: 1.0,
};
pub const BLACK: ColorF = ColorF {
    r: 0.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const SPRINGGREEN: ColorF = ColorF {
    r: 0.0,
    g: 1.0,
    b: 0.498039215686,
    a: 1.0,
};
pub const CRIMSON: ColorF = ColorF {
    r: 0.862745098039,
    g: 0.078431372549,
    b: 0.235294117647,
    a: 1.0,
};
pub const LIGHTSALMON: ColorF = ColorF {
    r: 1.0,
    g: 0.627450980392,
    b: 0.478431372549,
    a: 1.0,
};
pub const BROWN: ColorF = ColorF {
    r: 0.647058823529,
    g: 0.164705882353,
    b: 0.164705882353,
    a: 1.0,
};
pub const TURQUOISE: ColorF = ColorF {
    r: 0.250980392157,
    g: 0.878431372549,
    b: 0.81568627451,
    a: 1.0,
};
pub const OLIVEDRAB: ColorF = ColorF {
    r: 0.419607843137,
    g: 0.556862745098,
    b: 0.137254901961,
    a: 1.0,
};
pub const CYAN: ColorF = ColorF {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
pub const SILVER: ColorF = ColorF {
    r: 0.752941176471,
    g: 0.752941176471,
    b: 0.752941176471,
    a: 1.0,
};
pub const SKYBLUE: ColorF = ColorF {
    r: 0.529411764706,
    g: 0.807843137255,
    b: 0.921568627451,
    a: 1.0,
};
pub const GRAY: ColorF = ColorF {
    r: 0.501960784314,
    g: 0.501960784314,
    b: 0.501960784314,
    a: 1.0,
};
pub const DARKTURQUOISE: ColorF = ColorF {
    r: 0.0,
    g: 0.807843137255,
    b: 0.819607843137,
    a: 1.0,
};
pub const GOLDENROD: ColorF = ColorF {
    r: 0.854901960784,
    g: 0.647058823529,
    b: 0.125490196078,
    a: 1.0,
};
pub const DARKGREEN: ColorF = ColorF {
    r: 0.0,
    g: 0.392156862745,
    b: 0.0,
    a: 1.0,
};
pub const DARKVIOLET: ColorF = ColorF {
    r: 0.580392156863,
    g: 0.0,
    b: 0.827450980392,
    a: 1.0,
};
pub const DARKGRAY: ColorF = ColorF {
    r: 0.662745098039,
    g: 0.662745098039,
    b: 0.662745098039,
    a: 1.0,
};
pub const LIGHTPINK: ColorF = ColorF {
    r: 1.0,
    g: 0.713725490196,
    b: 0.756862745098,
    a: 1.0,
};
pub const TEAL: ColorF = ColorF {
    r: 0.0,
    g: 0.501960784314,
    b: 0.501960784314,
    a: 1.0,
};
pub const DARKMAGENTA: ColorF = ColorF {
    r: 0.545098039216,
    g: 0.0,
    b: 0.545098039216,
    a: 1.0,
};
pub const LIGHTGOLDENRODYELLOW: ColorF = ColorF {
    r: 0.980392156863,
    g: 0.980392156863,
    b: 0.823529411765,
    a: 1.0,
};
pub const LAVENDER: ColorF = ColorF {
    r: 0.901960784314,
    g: 0.901960784314,
    b: 0.980392156863,
    a: 1.0,
};
pub const YELLOWGREEN: ColorF = ColorF {
    r: 0.603921568627,
    g: 0.803921568627,
    b: 0.196078431373,
    a: 1.0,
};
pub const THISTLE: ColorF = ColorF {
    r: 0.847058823529,
    g: 0.749019607843,
    b: 0.847058823529,
    a: 1.0,
};
pub const VIOLET: ColorF = ColorF {
    r: 0.933333333333,
    g: 0.509803921569,
    b: 0.933333333333,
    a: 1.0,
};
pub const NAVY: ColorF = ColorF {
    r: 0.0,
    g: 0.0,
    b: 0.501960784314,
    a: 1.0,
};
pub const DIMGREY: ColorF = ColorF {
    r: 0.411764705882,
    g: 0.411764705882,
    b: 0.411764705882,
    a: 1.0,
};
pub const ORCHID: ColorF = ColorF {
    r: 0.854901960784,
    g: 0.439215686275,
    b: 0.839215686275,
    a: 1.0,
};
pub const BLUE: ColorF = ColorF {
    r: 0.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
pub const GHOSTWHITE: ColorF = ColorF {
    r: 0.972549019608,
    g: 0.972549019608,
    b: 1.0,
    a: 1.0,
};
pub const HONEYDEW: ColorF = ColorF {
    r: 0.941176470588,
    g: 1.0,
    b: 0.941176470588,
    a: 1.0,
};
pub const CORNFLOWERBLUE: ColorF = ColorF {
    r: 0.392156862745,
    g: 0.58431372549,
    b: 0.929411764706,
    a: 1.0,
};
pub const DARKBLUE: ColorF = ColorF {
    r: 0.0,
    g: 0.0,
    b: 0.545098039216,
    a: 1.0,
};
pub const DARKKHAKI: ColorF = ColorF {
    r: 0.741176470588,
    g: 0.717647058824,
    b: 0.419607843137,
    a: 1.0,
};
pub const MEDIUMPURPLE: ColorF = ColorF {
    r: 0.576470588235,
    g: 0.439215686275,
    b: 0.858823529412,
    a: 1.0,
};
pub const CORNSILK: ColorF = ColorF {
    r: 1.0,
    g: 0.972549019608,
    b: 0.862745098039,
    a: 1.0,
};
pub const RED: ColorF = ColorF {
    r: 1.0,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const BISQUE: ColorF = ColorF {
    r: 1.0,
    g: 0.894117647059,
    b: 0.76862745098,
    a: 1.0,
};
pub const SLATEGRAY: ColorF = ColorF {
    r: 0.439215686275,
    g: 0.501960784314,
    b: 0.564705882353,
    a: 1.0,
};
pub const DARKCYAN: ColorF = ColorF {
    r: 0.0,
    g: 0.545098039216,
    b: 0.545098039216,
    a: 1.0,
};
pub const KHAKI: ColorF = ColorF {
    r: 0.941176470588,
    g: 0.901960784314,
    b: 0.549019607843,
    a: 1.0,
};
pub const WHEAT: ColorF = ColorF {
    r: 0.960784313725,
    g: 0.870588235294,
    b: 0.701960784314,
    a: 1.0,
};
pub const DEEPSKYBLUE: ColorF = ColorF {
    r: 0.0,
    g: 0.749019607843,
    b: 1.0,
    a: 1.0,
};
pub const REBECCAPURPLE: ColorF = ColorF {
    r: 0.4,
    g: 0.2,
    b: 0.6,
    a: 1.0,
};
pub const DARKRED: ColorF = ColorF {
    r: 0.545098039216,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const STEELBLUE: ColorF = ColorF {
    r: 0.274509803922,
    g: 0.509803921569,
    b: 0.705882352941,
    a: 1.0,
};
pub const ALICEBLUE: ColorF = ColorF {
    r: 0.941176470588,
    g: 0.972549019608,
    b: 1.0,
    a: 1.0,
};
pub const LIGHTSLATEGREY: ColorF = ColorF {
    r: 0.466666666667,
    g: 0.533333333333,
    b: 0.6,
    a: 1.0,
};
pub const GAINSBORO: ColorF = ColorF {
    r: 0.862745098039,
    g: 0.862745098039,
    b: 0.862745098039,
    a: 1.0,
};
pub const MEDIUMTURQUOISE: ColorF = ColorF {
    r: 0.282352941176,
    g: 0.819607843137,
    b: 0.8,
    a: 1.0,
};
pub const FLORALWHITE: ColorF = ColorF {
    r: 1.0,
    g: 0.980392156863,
    b: 0.941176470588,
    a: 1.0,
};
pub const CORAL: ColorF = ColorF {
    r: 1.0,
    g: 0.498039215686,
    b: 0.313725490196,
    a: 1.0,
};
pub const PURPLE: ColorF = ColorF {
    r: 0.501960784314,
    g: 0.0,
    b: 0.501960784314,
    a: 1.0,
};
pub const LIGHTGREY: ColorF = ColorF {
    r: 0.827450980392,
    g: 0.827450980392,
    b: 0.827450980392,
    a: 1.0,
};
pub const LIGHTCYAN: ColorF = ColorF {
    r: 0.878431372549,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
pub const DARKSALMON: ColorF = ColorF {
    r: 0.913725490196,
    g: 0.588235294118,
    b: 0.478431372549,
    a: 1.0,
};
pub const BEIGE: ColorF = ColorF {
    r: 0.960784313725,
    g: 0.960784313725,
    b: 0.862745098039,
    a: 1.0,
};
pub const AZURE: ColorF = ColorF {
    r: 0.941176470588,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
pub const LIGHTSTEELBLUE: ColorF = ColorF {
    r: 0.690196078431,
    g: 0.76862745098,
    b: 0.870588235294,
    a: 1.0,
};
pub const OLDLACE: ColorF = ColorF {
    r: 0.992156862745,
    g: 0.960784313725,
    b: 0.901960784314,
    a: 1.0,
};
pub const GREENYELLOW: ColorF = ColorF {
    r: 0.678431372549,
    g: 1.0,
    b: 0.18431372549,
    a: 1.0,
};
pub const ROYALBLUE: ColorF = ColorF {
    r: 0.254901960784,
    g: 0.411764705882,
    b: 0.882352941176,
    a: 1.0,
};
pub const LIGHTSEAGREEN: ColorF = ColorF {
    r: 0.125490196078,
    g: 0.698039215686,
    b: 0.666666666667,
    a: 1.0,
};
pub const MISTYROSE: ColorF = ColorF {
    r: 1.0,
    g: 0.894117647059,
    b: 0.882352941176,
    a: 1.0,
};
pub const SIENNA: ColorF = ColorF {
    r: 0.627450980392,
    g: 0.321568627451,
    b: 0.176470588235,
    a: 1.0,
};
pub const LIGHTCORAL: ColorF = ColorF {
    r: 0.941176470588,
    g: 0.501960784314,
    b: 0.501960784314,
    a: 1.0,
};
pub const ORANGERED: ColorF = ColorF {
    r: 1.0,
    g: 0.270588235294,
    b: 0.0,
    a: 1.0,
};
pub const NAVAJOWHITE: ColorF = ColorF {
    r: 1.0,
    g: 0.870588235294,
    b: 0.678431372549,
    a: 1.0,
};
pub const LIME: ColorF = ColorF {
    r: 0.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};
pub const PALEGREEN: ColorF = ColorF {
    r: 0.596078431373,
    g: 0.98431372549,
    b: 0.596078431373,
    a: 1.0,
};
pub const BURLYWOOD: ColorF = ColorF {
    r: 0.870588235294,
    g: 0.721568627451,
    b: 0.529411764706,
    a: 1.0,
};
pub const SEASHELL: ColorF = ColorF {
    r: 1.0,
    g: 0.960784313725,
    b: 0.933333333333,
    a: 1.0,
};
pub const MEDIUMSPRINGGREEN: ColorF = ColorF {
    r: 0.0,
    g: 0.980392156863,
    b: 0.603921568627,
    a: 1.0,
};
pub const FUCHSIA: ColorF = ColorF {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
pub const PAPAYAWHIP: ColorF = ColorF {
    r: 1.0,
    g: 0.937254901961,
    b: 0.835294117647,
    a: 1.0,
};
pub const BLANCHEDALMOND: ColorF = ColorF {
    r: 1.0,
    g: 0.921568627451,
    b: 0.803921568627,
    a: 1.0,
};
pub const PERU: ColorF = ColorF {
    r: 0.803921568627,
    g: 0.521568627451,
    b: 0.247058823529,
    a: 1.0,
};
pub const AQUAMARINE: ColorF = ColorF {
    r: 0.498039215686,
    g: 1.0,
    b: 0.83137254902,
    a: 1.0,
};
pub const WHITE: ColorF = ColorF {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
pub const DARKSLATEGRAY: ColorF = ColorF {
    r: 0.18431372549,
    g: 0.309803921569,
    b: 0.309803921569,
    a: 1.0,
};
pub const TOMATO: ColorF = ColorF {
    r: 1.0,
    g: 0.388235294118,
    b: 0.278431372549,
    a: 1.0,
};
pub const IVORY: ColorF = ColorF {
    r: 1.0,
    g: 1.0,
    b: 0.941176470588,
    a: 1.0,
};
pub const DODGERBLUE: ColorF = ColorF {
    r: 0.117647058824,
    g: 0.564705882353,
    b: 1.0,
    a: 1.0,
};
pub const LEMONCHIFFON: ColorF = ColorF {
    r: 1.0,
    g: 0.980392156863,
    b: 0.803921568627,
    a: 1.0,
};
pub const CHOCOLATE: ColorF = ColorF {
    r: 0.823529411765,
    g: 0.411764705882,
    b: 0.117647058824,
    a: 1.0,
};
pub const ORANGE: ColorF = ColorF {
    r: 1.0,
    g: 0.647058823529,
    b: 0.0,
    a: 1.0,
};
pub const FORESTGREEN: ColorF = ColorF {
    r: 0.133333333333,
    g: 0.545098039216,
    b: 0.133333333333,
    a: 1.0,
};
pub const DARKGREY: ColorF = ColorF {
    r: 0.662745098039,
    g: 0.662745098039,
    b: 0.662745098039,
    a: 1.0,
};
pub const OLIVE: ColorF = ColorF {
    r: 0.501960784314,
    g: 0.501960784314,
    b: 0.0,
    a: 1.0,
};
pub const MINTCREAM: ColorF = ColorF {
    r: 0.960784313725,
    g: 1.0,
    b: 0.980392156863,
    a: 1.0,
};
pub const ANTIQUEWHITE: ColorF = ColorF {
    r: 0.980392156863,
    g: 0.921568627451,
    b: 0.843137254902,
    a: 1.0,
};
pub const DARKORANGE: ColorF = ColorF {
    r: 1.0,
    g: 0.549019607843,
    b: 0.0,
    a: 1.0,
};
pub const CADETBLUE: ColorF = ColorF {
    r: 0.372549019608,
    g: 0.619607843137,
    b: 0.627450980392,
    a: 1.0,
};
pub const MOCCASIN: ColorF = ColorF {
    r: 1.0,
    g: 0.894117647059,
    b: 0.709803921569,
    a: 1.0,
};
pub const LIMEGREEN: ColorF = ColorF {
    r: 0.196078431373,
    g: 0.803921568627,
    b: 0.196078431373,
    a: 1.0,
};
pub const SADDLEBROWN: ColorF = ColorF {
    r: 0.545098039216,
    g: 0.270588235294,
    b: 0.0745098039216,
    a: 1.0,
};
pub const GREY: ColorF = ColorF {
    r: 0.501960784314,
    g: 0.501960784314,
    b: 0.501960784314,
    a: 1.0,
};
pub const DARKSLATEBLUE: ColorF = ColorF {
    r: 0.282352941176,
    g: 0.239215686275,
    b: 0.545098039216,
    a: 1.0,
};
pub const LIGHTSKYBLUE: ColorF = ColorF {
    r: 0.529411764706,
    g: 0.807843137255,
    b: 0.980392156863,
    a: 1.0,
};
pub const DEEPPINK: ColorF = ColorF {
    r: 1.0,
    g: 0.078431372549,
    b: 0.576470588235,
    a: 1.0,
};
pub const PLUM: ColorF = ColorF {
    r: 0.866666666667,
    g: 0.627450980392,
    b: 0.866666666667,
    a: 1.0,
};
pub const AQUA: ColorF = ColorF {
    r: 0.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};
pub const DARKGOLDENROD: ColorF = ColorF {
    r: 0.721568627451,
    g: 0.525490196078,
    b: 0.043137254902,
    a: 1.0,
};
pub const MAROON: ColorF = ColorF {
    r: 0.501960784314,
    g: 0.0,
    b: 0.0,
    a: 1.0,
};
pub const SANDYBROWN: ColorF = ColorF {
    r: 0.956862745098,
    g: 0.643137254902,
    b: 0.376470588235,
    a: 1.0,
};
pub const MAGENTA: ColorF = ColorF {
    r: 1.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};
pub const TAN: ColorF = ColorF {
    r: 0.823529411765,
    g: 0.705882352941,
    b: 0.549019607843,
    a: 1.0,
};
pub const ROSYBROWN: ColorF = ColorF {
    r: 0.737254901961,
    g: 0.560784313725,
    b: 0.560784313725,
    a: 1.0,
};
pub const PINK: ColorF = ColorF {
    r: 1.0,
    g: 0.752941176471,
    b: 0.796078431373,
    a: 1.0,
};
pub const LIGHTBLUE: ColorF = ColorF {
    r: 0.678431372549,
    g: 0.847058823529,
    b: 0.901960784314,
    a: 1.0,
};
pub const PALEVIOLETRED: ColorF = ColorF {
    r: 0.858823529412,
    g: 0.439215686275,
    b: 0.576470588235,
    a: 1.0,
};
pub const MEDIUMSEAGREEN: ColorF = ColorF {
    r: 0.235294117647,
    g: 0.701960784314,
    b: 0.443137254902,
    a: 1.0,
};
pub const SLATEBLUE: ColorF = ColorF {
    r: 0.41568627451,
    g: 0.352941176471,
    b: 0.803921568627,
    a: 1.0,
};
pub const DIMGRAY: ColorF = ColorF {
    r: 0.411764705882,
    g: 0.411764705882,
    b: 0.411764705882,
    a: 1.0,
};
pub const POWDERBLUE: ColorF = ColorF {
    r: 0.690196078431,
    g: 0.878431372549,
    b: 0.901960784314,
    a: 1.0,
};
pub const SEAGREEN: ColorF = ColorF {
    r: 0.180392156863,
    g: 0.545098039216,
    b: 0.341176470588,
    a: 1.0,
};
pub const SNOW: ColorF = ColorF {
    r: 1.0,
    g: 0.980392156863,
    b: 0.980392156863,
    a: 1.0,
};
pub const MEDIUMBLUE: ColorF = ColorF {
    r: 0.0,
    g: 0.0,
    b: 0.803921568627,
    a: 1.0,
};
pub const MIDNIGHTBLUE: ColorF = ColorF {
    r: 0.0980392156863,
    g: 0.0980392156863,
    b: 0.439215686275,
    a: 1.0,
};
pub const PALETURQUOISE: ColorF = ColorF {
    r: 0.686274509804,
    g: 0.933333333333,
    b: 0.933333333333,
    a: 1.0,
};
pub const PALEGOLDENROD: ColorF = ColorF {
    r: 0.933333333333,
    g: 0.909803921569,
    b: 0.666666666667,
    a: 1.0,
};
pub const WHITESMOKE: ColorF = ColorF {
    r: 0.960784313725,
    g: 0.960784313725,
    b: 0.960784313725,
    a: 1.0,
};
pub const DARKORCHID: ColorF = ColorF {
    r: 0.6,
    g: 0.196078431373,
    b: 0.8,
    a: 1.0,
};
pub const SALMON: ColorF = ColorF {
    r: 0.980392156863,
    g: 0.501960784314,
    b: 0.447058823529,
    a: 1.0,
};
pub const LIGHTSLATEGRAY: ColorF = ColorF {
    r: 0.466666666667,
    g: 0.533333333333,
    b: 0.6,
    a: 1.0,
};
pub const LAWNGREEN: ColorF = ColorF {
    r: 0.486274509804,
    g: 0.988235294118,
    b: 0.0,
    a: 1.0,
};
pub const LIGHTGREEN: ColorF = ColorF {
    r: 0.564705882353,
    g: 0.933333333333,
    b: 0.564705882353,
    a: 1.0,
};
pub const LIGHTGRAY: ColorF = ColorF {
    r: 0.827450980392,
    g: 0.827450980392,
    b: 0.827450980392,
    a: 1.0,
};
pub const HOTPINK: ColorF = ColorF {
    r: 1.0,
    g: 0.411764705882,
    b: 0.705882352941,
    a: 1.0,
};
pub const LIGHTYELLOW: ColorF = ColorF {
    r: 1.0,
    g: 1.0,
    b: 0.878431372549,
    a: 1.0,
};
pub const LAVENDERBLUSH: ColorF = ColorF {
    r: 1.0,
    g: 0.941176470588,
    b: 0.960784313725,
    a: 1.0,
};
pub const LINEN: ColorF = ColorF {
    r: 0.980392156863,
    g: 0.941176470588,
    b: 0.901960784314,
    a: 1.0,
};
pub const MEDIUMAQUAMARINE: ColorF = ColorF {
    r: 0.4,
    g: 0.803921568627,
    b: 0.666666666667,
    a: 1.0,
};
pub const GREEN: ColorF = ColorF {
    r: 0.0,
    g: 0.501960784314,
    b: 0.0,
    a: 1.0,
};
pub const BLUEVIOLET: ColorF = ColorF {
    r: 0.541176470588,
    g: 0.16862745098,
    b: 0.886274509804,
    a: 1.0,
};
pub const PEACHPUFF: ColorF = ColorF {
    r: 1.0,
    g: 0.854901960784,
    b: 0.725490196078,
    a: 1.0,
};
