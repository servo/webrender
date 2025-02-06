// Copyright 2024 The Servo Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn precise_time_ns() -> u64 {
    platform::now()
}

#[cfg(all(unix, not(any(target_os = "macos", target_os = "ios"))))]
mod platform {
    use libc::timespec;

    #[allow(unsafe_code)]
    pub(super) fn now() -> u64 {
        // SAFETY: libc::timespec is zero initializable.
        let time = unsafe {
            let mut time: timespec = std::mem::zeroed();
            libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time);
            time
        };
        (time.tv_sec as u64) * 1000000000 + (time.tv_nsec as u64)
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod platform {
    use std::sync::LazyLock;

    use mach2::mach_time::{mach_absolute_time, mach_timebase_info};

    #[allow(unsafe_code)]
    fn timebase_info() -> &'static mach_timebase_info {
        static TIMEBASE_INFO: LazyLock<mach_timebase_info> = LazyLock::new(|| {
            let mut timebase_info = mach_timebase_info { numer: 0, denom: 0 };
            unsafe { mach_timebase_info(&mut timebase_info) };
            timebase_info
        });
        &TIMEBASE_INFO
    }

    #[allow(unsafe_code)]
    pub(super) fn now() -> u64 {
        let timebase_info = timebase_info();
        let absolute_time = unsafe { mach_absolute_time() };
        absolute_time * timebase_info.numer as u64 / timebase_info.denom as u64
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use std::sync::atomic::{AtomicU64, Ordering};

    use windows_sys::Win32::System::Performance::{
        QueryPerformanceCounter, QueryPerformanceFrequency,
    };

    /// The frequency of the value returned by `QueryPerformanceCounter` in counts per
    /// second. This is taken from the Rust source code at:
    /// <https://github.com/rust-lang/rust/blob/1a1cc050d8efc906ede39f444936ade1fdc9c6cb/library/std/src/sys/pal/windows/time.rs#L197>
    #[allow(unsafe_code)]
    fn frequency() -> i64 {
        // Either the cached result of `QueryPerformanceFrequency` or `0` for
        // uninitialized. Storing this as a single `AtomicU64` allows us to use
        // `Relaxed` operations, as we are only interested in the effects on a
        // single memory location.
        static FREQUENCY: AtomicU64 = AtomicU64::new(0);

        let cached = FREQUENCY.load(Ordering::Relaxed);
        // If a previous thread has filled in this global state, use that.
        if cached != 0 {
            return cached as i64;
        }
        // ... otherwise learn for ourselves ...
        let mut frequency = 0;
        let result = unsafe { QueryPerformanceFrequency(&mut frequency) };

        if result == 0 {
            return 0;
        }

        FREQUENCY.store(frequency as u64, Ordering::Relaxed);
        frequency
    }

    #[allow(unsafe_code)]
    /// Get the current instant value in nanoseconds.
    /// Originally from: <https://github.com/rust-lang/rust/blob/1a1cc050d8efc906ede39f444936ade1fdc9c6cb/library/std/src/sys/pal/windows/time.rs#L175>
    pub(super) fn now() -> u64 {
        let mut counter_value = 0;
        unsafe { QueryPerformanceCounter(&mut counter_value) };

        /// Computes (value*numer)/denom without overflow, as long as both
        /// (numer*denom) and the overall result fit into i64 (which is the case
        /// for our time conversions).
        /// Originally from: <https://github.com/rust-lang/rust/blob/1a1cc050d8efc906ede39f444936ade1fdc9c6cb/library/std/src/sys_common/mod.rs#L75>
        fn mul_div_u64(value: u64, numer: u64, denom: u64) -> u64 {
            let q = value / denom;
            let r = value % denom;
            // Decompose value as (value/denom*denom + value%denom),
            // substitute into (value*numer)/denom and simplify.
            // r < denom, so (denom*numer) is the upper bound of (r*numer)
            q * numer + r * numer / denom
        }

        static NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
        mul_div_u64(
            counter_value as u64,
            NANOSECONDS_PER_SECOND,
            frequency() as u64,
        )
    }
}

