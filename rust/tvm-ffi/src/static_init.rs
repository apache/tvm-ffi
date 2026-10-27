/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Static init blocks: code that runs when the program or library that holds it loads, as a C++
//! `TVM_FFI_STATIC_INIT_BLOCK` does. The blocks of a program or library are gathered at link
//! time, wherever in it they are, and a constructor runs them when it loads. Once Python loads a
//! library built on this crate, by any means, it sees what the blocks registered, as it sees the
//! registrations of a C++ library. A block that several libraries of a process have, as each
//! links its own copy of a crate they share, runs once. See
//! [`static_init_block!`](crate::static_init_block).

use std::ffi::c_void;
use std::panic::catch_unwind;
use std::sync::{Once, OnceLock};

use crate::function_internal::panic_to_error;
use crate::{Any, Error, Function, Result, INTERNAL_ERROR};

#[doc(hidden)]
pub use linkme;

/// Run code when the program or library that holds it loads, as a C++
/// `TVM_FFI_STATIC_INIT_BLOCK` does. The body is a block that returns [`Result<()>`](crate::Result):
/// it registers what the library defines, such as global functions, and stops at its first error.
///
/// ```
/// use tvm_ffi::{Function, Result};
///
/// fn add_one(x: i64) -> Result<i64> {
///     Ok(x + 1)
/// }
///
/// tvm_ffi::static_init_block! {
///     Function::register_global_typed("my_ext.add_one", add_one, "Add one to an integer.")?;
///     Ok(())
/// }
///
/// // Registered when the program loaded.
/// let add_one = Function::get_global("my_ext.add_one")?;
/// assert_eq!(i64::try_from(add_one.call_tuple((1i64,))?)?, 2);
/// # Ok::<(), tvm_ffi::Error>(())
/// ```
///
/// A block runs when its program or library loads: it adds a constructor, which the loader runs,
/// and the first constructor runs every block of the program or library, in the order of their
/// modules and lines. Python sees what the blocks of a library registered once it loads the
/// library, by any means, and the library stays loaded for the process, as the registrations keep
/// its code. An error cannot reach a caller when the library loads: it is printed, and
/// [`run_static_init_blocks`](crate::run_static_init_blocks) reports it again. Where the loader
/// does not run the constructor, call [`run_static_init_blocks`](crate::run_static_init_blocks)
/// first. A crate that a library depends on but never names is not linked into it, nor are its
/// blocks: name it, as `use my_crate as _;` does.
///
/// A block runs once per process. Every library built on the crate links its own copy of the
/// crates it uses, so a block of a crate that several libraries link is in each of them; the first
/// library that runs it registers what it defines, and the others skip it. A block of another
/// version of the crate, or whose text differs, is another block, and registering what it defines
/// again is an error. A change to the body of a function the block calls does not make another
/// block: rebuild every library that links a crate after changing it.
#[macro_export]
macro_rules! static_init_block {
    ($($body:tt)*) => {
        const _: () = {
            #[$crate::static_init::linkme::distributed_slice($crate::static_init::STATIC_INIT_BLOCKS)]
            #[linkme(crate = $crate::static_init::linkme)]
            static BLOCK: $crate::static_init::StaticInitBlock =
                $crate::static_init::StaticInitBlock::__new(
                    module_path!(),
                    line!(),
                    match option_env!("CARGO_PKG_VERSION") {
                        Some(version) => version,
                        None => "",
                    },
                    stringify!($($body)*),
                    || -> $crate::Result<()> { $($body)* },
                );
            #[used]
            #[cfg_attr(target_os = "linux", link_section = ".init_array")]
            #[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
            #[cfg_attr(windows, link_section = ".CRT$XCU")]
            static RUN_AT_LOAD: extern "C" fn() = $crate::static_init::__run_at_load;
        };
    };
}

/// A [`static_init_block!`](crate::static_init_block), gathered at link time.
#[doc(hidden)]
pub struct StaticInitBlock {
    /// The module and line of the block.
    module: &'static str,
    line: u32,
    /// The version of the block's crate and a hash of the block's text, which with the module and
    /// line identify the block in every library that links the crate.
    version: &'static str,
    text_hash: u64,
    body: fn() -> Result<()>,
    /// The outcome of the block, which runs once: the error's text if it failed.
    outcome: OnceLock<std::result::Result<(), std::string::String>>,
}

/// The [`static_init_block!`](crate::static_init_block)s of the program or library, gathered at
/// link time from the whole of it.
#[doc(hidden)]
#[linkme::distributed_slice]
pub static STATIC_INIT_BLOCKS: [StaticInitBlock];

impl StaticInitBlock {
    #[doc(hidden)]
    pub const fn __new(
        module: &'static str,
        line: u32,
        version: &'static str,
        text: &str,
        body: fn() -> Result<()>,
    ) -> Self {
        let text_hash = text_hash(text);
        Self {
            module,
            line,
            version,
            text_hash,
            body,
            outcome: OnceLock::new(),
        }
    }

    /// The identity of the block in the process, `module:line@version#hash`: the same in every
    /// library that links the block's crate, and another one if the crate version or the block's
    /// text differs.
    #[doc(hidden)]
    pub fn identity(&self) -> std::string::String {
        format!(
            "{}:{}@{}#{:016x}",
            self.module, self.line, self.version, self.text_hash
        )
    }

    /// The name of the global function that marks the block run in the process,
    /// `ffi.static_init_block.<identity>`. The libraries of a process share the global function
    /// table, and their statics are their own.
    #[doc(hidden)]
    pub fn marker(&self) -> std::string::String {
        format!("ffi.static_init_block.{}", self.identity())
    }

    /// Run the block if it has not run. The caller that runs a failing block gets its error;
    /// later ones get an error that names the block.
    #[doc(hidden)]
    pub fn run(&self) -> Result<()> {
        let mut failure = None;
        let outcome = self.outcome.get_or_init(|| {
            self.run_once().map_err(|error| {
                let text = format!("{}: {}", error.kind(), error.message());
                failure = Some(error);
                text
            })
        });
        match (failure, outcome) {
            (Some(error), _) => Err(error),
            (None, Ok(())) => Ok(()),
            (None, Err(text)) => {
                let message = format!("The static init block at {} failed: {text}", self.place());
                Err(Error::new(INTERNAL_ERROR, &message, ""))
            }
        }
    }

    /// Run the block unless another library of the process has run it. Every library links its
    /// own copy of the crates it is built from, so the same block can be in several, and what it
    /// defines is registered once, by the first library that runs it. The block is marked once it
    /// succeeds, so that one that fails fails in every library. A panic in the body is its error.
    fn run_once(&self) -> Result<()> {
        let marker = self.marker();
        if Function::get_global(&marker).is_err() {
            catch_unwind(self.body).unwrap_or_else(|payload| Err(panic_to_error(payload)))?;
            Function::register_global(&marker, Function::from_packed(|_| Ok(Any::default())))?;
        }
        Ok(())
    }

    /// Where the block is, for an error.
    fn place(&self) -> std::string::String {
        format!("{}:{}", self.module, self.line)
    }
}

/// The FNV-1a hash of `text` without its whitespace, which does not depend on how the compiler
/// spaces the tokens of a block it stringifies.
const fn text_hash(text: &str) -> u64 {
    let bytes = text.as_bytes();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_whitespace() {
            hash = (hash ^ bytes[i] as u64).wrapping_mul(0x0100_0000_01b3);
        }
        i += 1;
    }
    hash
}

/// Run every block of this program or library that has not run, in the order of their modules
/// and lines, and return those that failed, each with its first error.
fn run_blocks() -> Vec<(&'static StaticInitBlock, Error)> {
    let mut blocks: Vec<&StaticInitBlock> = STATIC_INIT_BLOCKS.iter().collect();
    blocks.sort_by_key(|block| (block.module, block.line));
    blocks
        .into_iter()
        .filter_map(|block| block.run().err().map(|e| (block, e)))
        .collect()
}

/// Run every [`static_init_block!`](crate::static_init_block) of this program or library that has
/// not run, in the order of their modules and lines; running again does nothing. Every block is
/// tried, and the error lists those that failed. A block that another library of the process has
/// run, from the same version of its crate and the same text, is skipped: libraries that link the
/// same crate register what its blocks define once.
///
/// The blocks run when their program or library loads, so this reports the errors of those that
/// failed, which could not reach a caller then: a library exports it, with
/// [`tvm_ffi_dll_export_typed_func!`](crate::tvm_ffi_dll_export_typed_func), for its loader to
/// call. Where the loader does not run the blocks, call it before using what they define.
pub fn run_static_init_blocks() -> Result<()> {
    let mut failures = run_blocks();
    if failures.len() <= 1 {
        return failures.pop().map_or(Ok(()), |(_, error)| Err(error));
    }
    let mut message = format!("{} static init blocks failed:", failures.len());
    for (block, error) in &failures {
        message.push_str(&format!(
            "\n  {}: {}: {}",
            block.place(),
            error.kind(),
            error.message()
        ));
    }
    Err(Error::new(failures[0].1.kind(), &message, ""))
}

/// The constructor of every [`static_init_block!`](crate::static_init_block), which the loader
/// runs when the program or library loads: the first one keeps the library loaded and runs every
/// block of it. An error cannot reach a caller here, so it is printed; [`run_static_init_blocks`]
/// reports it again.
#[doc(hidden)]
pub extern "C" fn __run_at_load() {
    static LOADED: Once = Once::new();
    LOADED.call_once(|| {
        let library = keep_loaded().unwrap_or_else(|| "static_init_block!".to_string());
        match catch_unwind(run_blocks) {
            Ok(failures) => {
                for (block, error) in failures {
                    eprintln!(
                        "{library}: the static init block at {} failed: {}: {}",
                        block.place(),
                        error.kind(),
                        error.message()
                    );
                }
            }
            Err(_) => eprintln!("{library}: running the static init blocks panicked"),
        }
    });
}

/// Keep the library that holds this code loaded for the rest of the process, as what its blocks
/// register keeps its code: unloading it, as `tvm_ffi.load_module` does when Python drops the
/// module, would leave the registrations dangling. Returns the path of the library.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn keep_loaded() -> Option<std::string::String> {
    use std::ffi::{c_char, c_int, CStr};

    #[repr(C)]
    struct DlInfo {
        fname: *const c_char,
        fbase: *mut c_void,
        sname: *const c_char,
        saddr: *mut c_void,
    }
    extern "C" {
        fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    }
    // RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE.
    #[cfg(target_os = "linux")]
    const FLAGS: c_int = 0x1 | 0x4 | 0x1000;
    #[cfg(target_os = "macos")]
    const FLAGS: c_int = 0x1 | 0x10 | 0x80;

    let null = std::ptr::null_mut();
    let mut info = DlInfo {
        fname: null as _,
        fbase: null,
        sname: null as _,
        saddr: null,
    };
    // SAFETY: `dladdr` fills `info` for an address of this library, whose path stays valid while
    // the library is loaded; `dlopen` of a loaded library only marks it, and its handle is kept.
    unsafe {
        if dladdr(keep_loaded as *const c_void, &mut info) == 0 || info.fname.is_null() {
            return None;
        }
        dlopen(info.fname, FLAGS);
        Some(CStr::from_ptr(info.fname).to_string_lossy().into_owned())
    }
}

/// Keep the library that holds this code loaded for the rest of the process, as what its blocks
/// register keeps its code. Returns the path of the library.
#[cfg(windows)]
fn keep_loaded() -> Option<std::string::String> {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;

    type Module = *mut c_void;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetModuleHandleExW(flags: u32, module_name: *const u16, module: *mut Module) -> i32;
        fn GetModuleFileNameW(module: Module, filename: *mut u16, size: u32) -> u32;
    }
    // GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: the module that
    // holds the address stays loaded until the process exits.
    const FLAGS: u32 = 0x1 | 0x4;

    let mut module: Module = std::ptr::null_mut();
    let mut path = vec![0u16; 32768];
    // SAFETY: `GetModuleHandleExW` takes an address of this library as the name, and
    // `GetModuleFileNameW` writes at most `path.len()` units into `path`.
    unsafe {
        let address = keep_loaded as *const c_void as *const u16;
        if GetModuleHandleExW(FLAGS, address, &mut module) == 0 || module.is_null() {
            return None;
        }
        let len = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
        if len == 0 {
            return None;
        }
        Some(
            OsString::from_wide(&path[..len.min(path.len())])
                .to_string_lossy()
                .into_owned(),
        )
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn keep_loaded() -> Option<std::string::String> {
    None
}
