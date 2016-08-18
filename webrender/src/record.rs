use bincode::serde::serialize;
use bincode;
use std::fs::OpenOptions;
use std::io::Write;
use webrender_traits::ApiMsg;
use byteorder::{LittleEndian, WriteBytesExt};

pub fn write_data(frame:u32, data: &Vec<u8>){
       let filename = format!("record/frame_{}.bin", frame);
       let mut file = OpenOptions::new().append(true).create(true).open(filename).unwrap();
       file.write_u32::<LittleEndian>(data.len() as u32).ok();
       file.write(data).ok(); 
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
                       let buff = serialize(msg, bincode::SizeLimit::Infinite).unwrap();
                       write_data(frame, &buff)
               }
               _ => {}
       }
}

