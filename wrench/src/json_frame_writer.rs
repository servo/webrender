extern crate yaml_rust;

use app_units::Au;
use euclid::{Point2D, Size2D, Rect, Matrix4D};
use image::{ColorType, save_buffer};
use std::borrow::BorrowMut;
use std::cell::Cell;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fs;
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::slice;
use std::path::{Path, PathBuf};
use webrender;
use webrender_traits::*;
use serde_json;

use super::CURRENT_FRAME_NUMBER;

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

        JsonFrameWriter {
            frame_base: path.to_owned(),
            rsrc_base: rsrc_base,
            next_rsrc_num: 1,
            images: HashMap::new(),
            fonts: HashMap::new(),

            dl_descriptor: None,
            aux_descriptor: None,

            last_frame_written: u32::max_value(),
        }
    }

    pub fn begin_write_root_display_list(&mut self,
                                         _: &ColorF,
                                         _: &Epoch,
                                         _: &PipelineId,
                                         _: &Size2D<f32>,
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

        let mut auxiliary_data = Cursor::new(&data[4..]);

        let mut built_display_list_data = vec![0; dl_desc.size()];
        let mut aux_list_data = vec![0; aux_desc.size()];

        auxiliary_data.read_exact(&mut built_display_list_data[..]).unwrap();
        auxiliary_data.read_exact(&mut aux_list_data[..]).unwrap();

        let dl = BuiltDisplayList::from_data(built_display_list_data, dl_desc);
        let aux = AuxiliaryLists::from_data(aux_list_data, aux_desc);

        let mut frame_file_name = self.frame_base.clone();
        let current_shown_frame = unsafe { CURRENT_FRAME_NUMBER };
        frame_file_name.push(format!("frame-{}.yaml", current_shown_frame));

        let mut file = File::create(&frame_file_name).unwrap();

        let items: Vec<&DisplayItem> = dl.all_display_items().iter().collect();
        let s = serde_json::to_string_pretty(&items).unwrap();
        file.write_all(&s.into_bytes()).unwrap();
        file.write_all(b"\n").unwrap();
    }

    fn next_rsrc_paths(counter: &mut u32, base_path: &Path, base: &str, ext: &str) -> (PathBuf, PathBuf) {
        let mut path_file = base_path.to_owned();
        let mut path = PathBuf::from("res");

        let fstr = format!("{}-{}.{}", base, counter, ext);
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
        let bytes = data.bytes.take().unwrap();
        let (path_file, path) = Self::next_rsrc_paths(&mut self.next_rsrc_num, &self.rsrc_base, "img", "png");

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

            &ApiMsg::AddImage(ref key, width, height, stride,
                              format, ref data) => {
                let stride = if let Some(stride) = stride {
                    stride
                } else {
                    match format {
                        ImageFormat::A8 => width,
                        ImageFormat::RGBA8 | ImageFormat::RGB8 => width*4,
                        ImageFormat::RGBAF32 => width*16,
                        _ => panic!("Invalid image format"),
                    }
                };
                let bytes = match data {
                    &ImageData::Raw(ref v) => { (**v).clone() }
                    &ImageData::External(_) => { return; }
                };
                self.images.insert(*key, CachedImage {
                    width: width, height: height, stride: stride,
                    format: format,
                    bytes: Some(bytes),
                    path: None,
                });
            }

            &ApiMsg::UpdateImage(ref key, width, height,
                                 format, ref bytes) => {
                if let Some(ref mut data) = self.images.get_mut(key) {
                    assert!(data.width == width);
                    assert!(data.height == height);
                    assert!(data.format == format);

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
                                        ref auxiliary_lists) => {
                self.begin_write_root_display_list(background_color, epoch, pipeline_id,
                                                   viewport_size, display_list, auxiliary_lists);
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
