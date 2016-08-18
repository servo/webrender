use bincode;
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::OpenOptions;
use std::io::Write;
use webrender_traits::ApiMsg;

pub fn write_data(frame_counter:u32, auxiliary_data: &Vec<u8>){
    let filename = format!("record/frame_{}.bin", frame_counter);   
    let mut file = OpenOptions::new().append(true).open(filename).unwrap();
    file.write_u32::<LittleEndian>(auxiliary_data.len() as u32).ok();
    file.write(&auxiliary_data).ok();
}


pub fn write_msg(frame:u32, msg: &ApiMsg){
    match msg{
        ref msg @ &ApiMsg::AddRawFont(..) | 
            ref msg @ &ApiMsg::AddNativeFont(..) |
            ref msg @ &ApiMsg::AddImage(..) |
            ref msg @ &ApiMsg::SetRootPipeline(..) |
            ref msg @ &ApiMsg::UpdateImage(..) |
            ref msg @ &ApiMsg::Scroll(..)|
            ref msg @ &ApiMsg::TickScrollingBounce|
            ref msg @ &ApiMsg::DeleteImage(..)|
            ref msg @ &ApiMsg::SetRootStackingContext(..) =>{
                let filename = format!("record/frame_{}.bin", frame);
                let mut file = OpenOptions::new().append(true).create(true).open(filename).unwrap();
                let buff = bincode::serde::serialize(msg, bincode::SizeLimit::Infinite).unwrap();
                file.write_u32::<LittleEndian>(buff.len() as u32).unwrap();
                file.write(&buff).unwrap();
            }
        _ => {}
    }
}
