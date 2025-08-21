use std::{path::PathBuf};
use api::{NativeFontHandle};
use memmap2::Mmap;
use skrifa::raw::{FileRef};

pub(crate) struct SimpleFontHandle {
    pub path: PathBuf,
    pub index: u32,
    pub data: Mmap,
}

impl SimpleFontHandle {
    pub(crate) fn new(path: PathBuf, index: u32) -> Self {
        let file = std::fs::File::open(&path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        Self {
            path,
            index,
            data: mapped,
        }
    }
}

impl From<NativeFontHandle> for SimpleFontHandle {
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();

        SimpleFontHandle {
            path: native.path,
            index: native.index,
            data: mapped,
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn from(native: NativeFontHandle) -> Self {
        let file = std::fs::File::open(&native.path).unwrap();
        let mapped = unsafe { Mmap::map(&file) }.unwrap();
        let font_file = FileRef::new(&mapped).unwrap();
        let index = ttc_index_from_postscript_name(font_file, &native.name);

        SimpleFontHandle {
            path: PathBuf::from(native.path),
            index,
            data: mapped,
        }
    }
}

/// CoreText font enumaration gives us a postscript name rather than an index.
/// This functions maps from postscript name to index
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn ttc_index_from_postscript_name(font_file: FileRef<'_>, postscript_name: &str) -> u32 {
    use skrifa::raw::{FileRef, TableProvider as _};
    use skrifa::raw::types::NameId;

    let index = match font_file {
        FileRef::Font(_) => 0,
        FileRef::Collection(collection) => 'idx: {
            for i in 0..collection.len() {
                let font = collection.get(i).unwrap();
                let name_table = font.name().unwrap();
                if name_table
                    .name_record()
                    .iter()
                    .filter(|record| record.name_id() == NameId::POSTSCRIPT_NAME)
                    .any(|record| {
                        record
                            .string(name_table.string_data())
                            .unwrap()
                            .chars()
                            .eq(postscript_name.chars())
                    })
                {
                    break 'idx i;
                }
            }

            panic!(
                "Font with postscript_name {} not found in collection",
                postscript_name
            );
        }
    };

    index
}
