/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// the json code is largely unfinished; allow these to silence a bunch of warnings
#![allow(unused_variables)]
#![allow(dead_code)]

use image::{ColorType, save_buffer};
use premultiply::unpremultiply;
use serde_json;
use std::borrow::BorrowMut;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{fmt, fs};
use super::CURRENT_FRAME_NUMBER;
use time;
use webrender;
use webrender_traits::*;

enum CachedFont {
    Native(NativeFontHandle),
    Raw(Option<Vec<u8>>, Option<PathBuf>),
}

struct CachedImage {
    width: u32,
    height: u32,
    stride: u32,
    format: ImageFormat,
    bytes: Option<Vec<u8>>,
    path: Option<PathBuf>,
}

pub struct JsonFrameWriter {
    frame_base: PathBuf,
    rsrc_base: PathBuf,
    rsrc_prefix: String,
    next_rsrc_num: u32,
    images: HashMap<ImageKey, CachedImage>,
    fonts: HashMap<FontKey, CachedFont>,

    last_frame_written: u32,

    dl_descriptor: Option<BuiltDisplayListDescriptor>,
    aux_descriptor: Option<AuxiliaryListsDescriptor>,
}

impl JsonFrameWriter {
    pub fn new(path: &Path) -> JsonFrameWriter {
        let mut rsrc_base = path.to_owned();
        rsrc_base.push("res");
        fs::create_dir_all(&rsrc_base).ok();

        let rsrc_prefix = format!("{}", time::get_time().sec);

        JsonFrameWriter {
            frame_base: path.to_owned(),
            rsrc_base: rsrc_base,
            rsrc_prefix: rsrc_prefix,
            next_rsrc_num: 1,
            images: HashMap::new(),
            fonts: HashMap::new(),

            dl_descriptor: None,
            aux_descriptor: None,

            last_frame_written: u32::max_value(),
        }
    }

    pub fn begin_write_root_display_list(&mut self,
                                         _: &Option<ColorF>,
                                         _: &Epoch,
                                         _: &PipelineId,
                                         _: &LayoutSize,
                                         display_list: &BuiltDisplayListDescriptor,
                                         auxiliary_lists: &AuxiliaryListsDescriptor)
    {
        unsafe {
            if CURRENT_FRAME_NUMBER == self.last_frame_written {
                return;
            }
            self.last_frame_written = CURRENT_FRAME_NUMBER;
        }

        self.dl_descriptor = Some(display_list.clone());
        self.aux_descriptor = Some(auxiliary_lists.clone());
    }

    pub fn finish_write_root_display_list(&mut self,
                                          frame: u32,
                                          data: &[u8])
    {
        let dl_desc = self.dl_descriptor.take().unwrap();
        let aux_desc = self.aux_descriptor.take().unwrap();

        assert_eq!(data.len(), dl_desc.size() + aux_desc.size() + 4);

        // there's a 4 byte epoch header that we skip
        let dl_data = data[4..dl_desc.size() + 4].to_vec();
        let aux_data = data[dl_desc.size() + 4..].to_vec();

        let dl = BuiltDisplayList::from_data(dl_data, dl_desc);
        let aux = AuxiliaryLists::from_data(aux_data, aux_desc);

        let mut frame_file_name = self.frame_base.clone();
        let current_shown_frame = unsafe { CURRENT_FRAME_NUMBER };
        frame_file_name.push(format!("frame-{}.json", current_shown_frame));

        let mut file = File::create(&frame_file_name).unwrap();

        let items: Vec<&DisplayItem> = dl.all_display_items().iter().collect();
        let s = serde_json::to_string_pretty(&items).unwrap();
        file.write_all(&s.into_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
    }

    fn next_rsrc_paths(prefix: &str, counter: &mut u32, base_path: &Path, base: &str, ext: &str) -> (PathBuf, PathBuf) {
        let mut path_file = base_path.to_owned();
        let mut path = PathBuf::from("res");

        let fstr = format!("{}-{}-{}.{}", prefix, base, counter, ext);
        path_file.push(&fstr);
        path.push(&fstr);

        *counter += 1;

        (path_file, path)
    }

    fn path_for_image(&mut self, key: &ImageKey) -> Option<PathBuf> {
        if let Some(ref mut data) = self.images.get_mut(&key) {
            if data.path.is_some() {
                return data.path.clone();
            }
        } else {
            return None;
        };

        // Remove the data to munge it
        let mut data = self.images.remove(&key).unwrap();
        let mut bytes = data.bytes.take().unwrap();
        let (path_file, path) = Self::next_rsrc_paths(&self.rsrc_prefix,
                                                      &mut self.next_rsrc_num,
                                                      &self.rsrc_base,
                                                      "img",
                                                      "png");

        let ok = match data.format {
            ImageFormat::RGB8 => {
                if data.stride == data.width * 3 {
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::RGB(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            ImageFormat::RGBA8 => {
                if data.stride == data.width * 4 {
                    unpremultiply(bytes.as_mut_slice());
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::RGBA(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            ImageFormat::A8 => {
                if data.stride == data.width {
                    save_buffer(&path_file, &bytes, data.width, data.height, ColorType::Gray(8)).unwrap();
                    true
                } else {
                    false
                }
            }
            _ => { false }
        };

        if !ok {
            println!("Failed to write image with format {:?}, dimensions {}x{}, stride {}",
                     data.format, data.width, data.height, data.stride);
            return None;
        }

        data.path = Some(path.clone());
        // put it back
        self.images.insert(*key, data);
        Some(path)
    }
}

impl fmt::Debug for JsonFrameWriter {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "JsonFrameWriter")
    }
}

impl webrender::ApiRecordingReceiver for JsonFrameWriter {
    fn write_msg(&mut self, _: u32, msg: &ApiMsg) {
        match msg {
            &ApiMsg::SetRootPipeline(..) |
            &ApiMsg::Scroll(..) |
            &ApiMsg::TickScrollingBounce |
            &ApiMsg::WebGLCommand(..) => {
            }

            &ApiMsg::AddRawFont(ref key, ref bytes) => {
                self.fonts.insert(*key, CachedFont::Raw(Some(bytes.clone()), None));
            }

            &ApiMsg::AddNativeFont(ref key, ref native_font_handle) => {
                self.fonts.insert(*key, CachedFont::Native(native_font_handle.clone()));
            }

            &ApiMsg::AddImage(ref key, ref descriptor, ref data, _) => {
                let stride = descriptor.stride.unwrap_or(
                    descriptor.width * descriptor.format.bytes_per_pixel().unwrap()
                );
                let bytes = match data {
                    &ImageData::Raw(ref v) => { (**v).clone() }
                    &ImageData::ExternalHandle(_) => { return; }
                    &ImageData::ExternalBuffer(_) => { return; }
                    &ImageData::Blob(_) => { return; }
                };
                self.images.insert(*key, CachedImage {
                    width: descriptor.width,
                    height: descriptor.height,
                    stride: stride,
                    format: descriptor.format,
                    bytes: Some(bytes),
                    path: None,
                });
            }

            &ApiMsg::UpdateImage(ref key, descriptor, ref bytes, _dirty_rect) => {
                if let Some(ref mut data) = self.images.get_mut(key) {
                    assert!(data.width == descriptor.width);
                    assert!(data.height == descriptor.height);
                    assert!(data.format == descriptor.format);

                    *data.path.borrow_mut() = None;
                    *data.bytes.borrow_mut() = Some(bytes.clone());
                }
            }

            &ApiMsg::DeleteImage(ref key) => {
                self.images.remove(key);
            }

            &ApiMsg::SetRootDisplayList(ref background_color,
                                        ref epoch,
                                        ref pipeline_id,
                                        ref viewport_size,
                                        ref display_list,
                                        ref auxiliary_lists,
                                        _preserve_frame_state) => {
                self.begin_write_root_display_list(background_color,
                                                   epoch,
                                                   pipeline_id,
                                                   viewport_size,
                                                   display_list,
                                                   auxiliary_lists);
            }
            _ => {}
        }
    }

    fn write_payload(&mut self, frame: u32, data: &[u8]) {
        if self.dl_descriptor.is_some() {
            self.finish_write_root_display_list(frame, data);
        }
    }
}
