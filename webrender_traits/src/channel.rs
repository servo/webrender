/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use byteorder::{LittleEndian, WriteBytesExt, ReadBytesExt};
use std::io::{Cursor, Read};
use api::Epoch;

#[derive(Clone)]
pub struct Payload {
    pub epoch: Epoch,
    pub display_list_data: Vec<u8>,
    pub auxiliary_lists_data: Vec<u8>
}

impl Payload {
    pub fn to_data(&self) -> Vec<u8> {
        let mut data = vec![];
        data.write_u32::<LittleEndian>(self.epoch.0).unwrap();
        data.write_u64::<LittleEndian>(self.display_list_data.len() as u64).unwrap();
        data.extend_from_slice(&self.display_list_data);
        data.write_u64::<LittleEndian>(self.auxiliary_lists_data.len() as u64).unwrap();
        data.extend_from_slice(&self.auxiliary_lists_data);
        data
    }
    pub fn from_data(data: Vec<u8>) -> Payload {
        let mut payload_reader = Cursor::new(&data[..]);
        let epoch = Epoch(payload_reader.read_u32::<LittleEndian>().unwrap());

        let dl_size = payload_reader.read_u64::<LittleEndian>().unwrap() as usize;
        let mut built_display_list_data = vec![0; dl_size];
        payload_reader.read_exact(&mut built_display_list_data[..]).unwrap();

        let aux_size = payload_reader.read_u64::<LittleEndian>().unwrap() as usize;
        let mut auxiliary_lists_data = vec![0; aux_size];
        payload_reader.read_exact(&mut auxiliary_lists_data[..]).unwrap();

        Payload{ epoch: epoch, display_list_data: built_display_list_data,
            auxiliary_lists_data: auxiliary_lists_data }
    }
}


// A helper to handle the interface difference between
// IpcBytesSender and Sender<Vec<u8>>
pub trait PayloadSenderHelperMethods {
    fn send(&self, data: Payload) -> Result<(), Error>;
}
pub trait PayloadReceiverHelperMethods {
    fn recv(&self) -> Result<Payload, Error>;
}

#[cfg(not(feature = "ipc"))]
include!("channel_mpsc.rs");

#[cfg(feature = "ipc")]
include!("channel_ipc.rs");
