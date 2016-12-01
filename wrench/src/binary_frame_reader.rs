/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use clap;
use std::mem;
use std::any::TypeId;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process;
use bincode::serde::deserialize;
use byteorder::{LittleEndian, ReadBytesExt};
use webrender::WEBRENDER_RECORDING_HEADER;
use webrender_traits::ApiMsg;
use wrench::{Wrench, WrenchThing};

#[derive(Clone)]
enum Item {
    Message(ApiMsg),
    Data(Vec<u8>),
}

pub struct BinaryFrameReader {
    file_base: PathBuf,
    is_dir: bool,

    skip_uploads: bool,
    replay_api: bool,
    play_through: bool,

    frame_data: Vec<Item>,
    frame_num: u32,
    frame_built: bool,

    check_apimsg: Option<bool>,
}

impl BinaryFrameReader {
    pub fn new(file_path: &Path) -> BinaryFrameReader {
        BinaryFrameReader {
            file_base: file_path.to_owned(),
            is_dir: file_path.is_dir(),

            skip_uploads: false,
            replay_api: false,
            play_through: false,

            frame_data: vec![],
            frame_num: 0,
            frame_built: false,

            check_apimsg: None,
        }
    }

    pub fn new_from_args(args: &clap::ArgMatches) -> BinaryFrameReader {
        let bin_file = args.value_of("INPUT").map(|s| PathBuf::from(s)).unwrap();
        let mut r = BinaryFrameReader::new(&bin_file);
        r.skip_uploads = args.is_present("skip-uploads");
        r.replay_api = args.is_present("api");
        r.play_through = args.is_present("play");
        r
    }

    // FIXME I don't think we can skip uploads without also skipping
    // payload (I think? Unused payload ranges may also be ignored.). But
    // either way we'd need to track image updates and deletions -- if we
    // delete an image, we can't go back to a previous frame.
    //
    // We could probably introduce a mode where going backwards replays all
    // frames up until that frame, so that we know we can be accurate.
    fn should_skip_upload_msg(&self, msg: &ApiMsg) -> bool {
        if !self.skip_uploads {
            return false;
        }

        match msg {
            &ApiMsg::AddRawFont(..) |
            &ApiMsg::AddNativeFont(..) |
            &ApiMsg::AddImage(..) |
            &ApiMsg::UpdateImage(..) |
            &ApiMsg::DeleteImage(..) => {
                true
            }
            _ => {
                false
            }
        }
    }

    fn frame_exists(&self, frame_num: u32) -> bool {
        let mut file_name = self.file_base.clone();
        file_name.push(format!("frame_{}.bin", frame_num));
        file_name.exists()
    }
}

impl WrenchThing for BinaryFrameReader {
    fn do_frame(&mut self, wrench: &mut Wrench) -> u32 {
        let first_time = !self.frame_built;
        if first_time {
            wrench.set_title(&format!("frame {}", self.frame_num));

            // TODO mmap instead of read
            let mut file = if self.is_dir {
                let mut file_name = self.file_base.clone();
                file_name.push(format!("frame_{}.bin", self.frame_num));
                File::open(&file_name).expect("Frame file not found!")
            } else {
                File::open(&self.file_base).expect("Frame file not found!")
            };

            let check_apimsg = if let Some(check_apimsg) = self.check_apimsg {
                check_apimsg
            } else {
                let header = file.read_u64::<LittleEndian>().unwrap();
                let check_apimsg = header == WEBRENDER_RECORDING_HEADER;
                if !check_apimsg {
                    println!("Note: Binary file is old-style recording without ApiMsg TypeId");
                }

                // reset the file back to the start
                file.seek(SeekFrom::Start(0)).ok();
                self.check_apimsg = Some(check_apimsg);
                check_apimsg
            };

            if check_apimsg {
                let apimsg_type_id = unsafe {
                    assert!(mem::size_of::<TypeId>() == mem::size_of::<u64>());
                    mem::transmute::<TypeId, u64>(TypeId::of::<ApiMsg>())
                };

                let header = file.read_u64::<LittleEndian>().unwrap();
                assert!(header == WEBRENDER_RECORDING_HEADER);

                let written_apimsg_type_id = file.read_u64::<LittleEndian>().unwrap();
                if written_apimsg_type_id != apimsg_type_id {
                    println!("Binary file ApiMsg enum type mismatch: expected 0x{:x}, found 0x{:x}", apimsg_type_id, written_apimsg_type_id);
                }
            }

            self.frame_data.clear();
            while let Ok(mut len) = file.read_u32::<LittleEndian>() {
                if len > 0 {
                    let mut buffer = vec![0; len as usize];
                    file.read_exact(&mut buffer).unwrap();
                    let msg = deserialize(&buffer).unwrap();
                    self.frame_data.push(Item::Message(msg));
                } else {
                    len = file.read_u32::<LittleEndian>().unwrap();
                    let mut buffer = vec![0; len as usize];
                    file.read_exact(&mut buffer).unwrap();
                    self.frame_data.push(Item::Data(buffer));
                }
            }

            self.frame_built = true;
        }

        if first_time || self.replay_api {
            let mut seen_set_root_dl = false;
            wrench.begin_frame();
            let frame_items = self.frame_data.clone();
            for item in frame_items {
                match item {
                    Item::Message(msg) => {
                        if !self.should_skip_upload_msg(&msg) {
                            match &msg {
                                &ApiMsg::SetRootDisplayList(..) => { seen_set_root_dl = true; }
                                _ => { }
                            }
                            wrench.api.api_sender.send(msg).unwrap();
                        }
                    }
                    Item::Data(buf) => {
                        wrench.api.payload_sender.send(buf).unwrap();
                    }
                }
            }
            if !seen_set_root_dl && self.frame_num != 0 {
                println!("Frame {} didn't contain a SetRootDisplayList message!", self.frame_num);
                wrench.refresh();
            }
        } else if self.play_through {
            if !self.frame_exists(self.frame_num + 1) {
                process::exit(0);
            }
            self.next_frame();
            self.do_frame(wrench);
        } else {
            wrench.refresh();
        }

        // Frame 0 is.. weird.  It's always empty, but we can't skip it
        // or everything else breaks.
        if self.frame_num == 0 {
            self.next_frame();
            self.do_frame(wrench)
        } else {
            self.frame_num
        }
    }

    // note that we don't loop here; we could, but that'll require
    // some tracking to avoid reuploading resources every run.  We
    // sort of try in should_skip_upload_msg, but this needs work.
    fn next_frame(&mut self) {
        if self.frame_exists(self.frame_num + 1) {
            self.frame_num += 1;
            self.frame_built = false;
        }
    }

    fn prev_frame(&mut self) {
        if self.frame_num > 0 {
            self.frame_num -= 1;
            self.frame_built = false;
        }
    }
}
