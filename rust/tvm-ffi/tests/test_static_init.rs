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

//! The static init blocks of this test binary run when it loads, before the tests. The blocks of
//! another library are simulated with [`StaticInitBlock`]s of the same identity.

use std::sync::atomic::{AtomicUsize, Ordering};

use tvm_ffi::static_init::{StaticInitBlock, STATIC_INIT_BLOCKS};
use tvm_ffi::{run_static_init_blocks, Error, Function, Result, VALUE_ERROR};

fn add_one(x: i64) -> Result<i64> {
    Ok(x + 1)
}

tvm_ffi::static_init_block! {
    Function::register_global_typed("test_static_init.add_one", add_one, "Add one.")?;
}

// A block may also end in the `Result<()>` it returns.
mod elsewhere {
    use super::*;

    tvm_ffi::static_init_block! {
        Function::register_global_typed("test_static_init.elsewhere", || -> Result<i64> { Ok(7) }, "")?;
        Ok(())
    }
}

// A block that returns early, and one that ends in `return`, which `!` types.
mod returning {
    use super::*;

    fn body() -> Result<i64> {
        Ok(8)
    }

    tvm_ffi::static_init_block! {
        Function::register_global_typed("test_static_init.returning", body, "")?;
        if Function::get_global("test_static_init.returning").is_ok() {
            return Ok(());
        }
        Function::register_global_typed("test_static_init.not_reached", body, "")?;
    }

    tvm_ffi::static_init_block! {
        Function::register_global_typed("test_static_init.returned", body, "")?;
        return Ok(())
    }
}

// Blocks that fail, which `run_static_init_blocks` reports together while it runs the others: a
// block stops at its first error, and a panic is an error.
mod failing {
    use super::*;

    tvm_ffi::static_init_block! {
        Function::register_global_typed("test_static_init.twice", add_one, "")?;
        Function::register_global_typed("test_static_init.twice", add_one, "")?;
        Function::register_global_typed("test_static_init.after_broken", add_one, "")?;
    }

    tvm_ffi::static_init_block! {
        Err(Error::new(VALUE_ERROR, "the second block fails", ""))
    }

    tvm_ffi::static_init_block! {
        panic!("the third block panics")
    }
}

fn call_i64(name: &str) -> Result<i64> {
    i64::try_from(Function::get_global(name)?.call_tuple(())?)
}

#[test]
fn blocks_run_when_the_program_loads() {
    let add_one = Function::get_global("test_static_init.add_one").unwrap();
    assert_eq!(
        i64::try_from(add_one.call_tuple((1i64,)).unwrap()).unwrap(),
        2
    );
    assert_eq!(call_i64("test_static_init.elsewhere").unwrap(), 7);
    // A block's own items are in scope, not shadowed by those of the macro.
    assert_eq!(call_i64("test_static_init.returning").unwrap(), 8);
    assert_eq!(call_i64("test_static_init.returned").unwrap(), 8);
    assert!(Function::get_global("test_static_init.not_reached").is_err());

    // The block that failed registered what came before its error, and stopped there.
    assert!(Function::get_global("test_static_init.twice").is_ok());
    assert!(Function::get_global("test_static_init.after_broken").is_err());

    // Every block of the binary is gathered, wherever it is.
    let blocks = STATIC_INIT_BLOCKS
        .iter()
        .filter(|block| block.identity().starts_with("test_static_init"))
        .count();
    assert_eq!(blocks, 7);
}

#[test]
fn blocks_that_fail_are_reported_together() {
    let error = run_static_init_blocks().unwrap_err();
    let message = error.message();
    assert!(
        message.starts_with("3 static init blocks failed:"),
        "{message}"
    );
    assert!(
        message.contains("\n  test_static_init::failing:"),
        "{message}"
    );
    assert!(
        message.contains(
            "RuntimeError: Global Function `test_static_init.twice` is already registered"
        ),
        "{message}"
    );
    assert!(
        message.contains("ValueError: the second block fails"),
        "{message}"
    );
    assert!(
        message.contains("InternalError: panicked: the third block panics"),
        "{message}"
    );

    // A failed block does not run again, and is reported again.
    assert_eq!(run_static_init_blocks().unwrap_err().message(), message);
}

#[test]
fn a_block_another_library_ran_is_skipped() {
    static RUNS: AtomicUsize = AtomicUsize::new(0);
    fn body() -> Result<()> {
        RUNS.fetch_add(1, Ordering::SeqCst);
        Function::register_global_typed("test_static_init.shared", add_one, "")?;
        Ok(())
    }
    // One block in two libraries, as one of a crate both link.
    let text =
        r#"Function::register_global_typed("test_static_init.shared", add_one, "")?; Ok(())"#;
    let block = || StaticInitBlock::__new("ext::shared", 3, "0.1.0", text, body);
    let (first, second) = (block(), block());
    assert_eq!(first.marker(), second.marker());
    assert!(first
        .marker()
        .starts_with("ffi.static_init_block.ext::shared:3@0.1.0#"));

    // The first library runs the block and marks it; the second skips it, where registering the
    // name again would fail.
    first.run().unwrap();
    assert!(Function::get_global(&first.marker()).is_ok());
    assert_eq!(RUNS.load(Ordering::SeqCst), 1);
    second.run().unwrap();
    assert_eq!(RUNS.load(Ordering::SeqCst), 1);

    // Another version of the crate is another block, whose name is already registered. Its library
    // gets the error, and a later caller an error that names the block.
    let other = StaticInitBlock::__new("ext::shared", 3, "0.2.0", text, body);
    let error = other.run().unwrap_err();
    assert!(error
        .message()
        .contains("`test_static_init.shared` is already registered"));
    assert_eq!(RUNS.load(Ordering::SeqCst), 2);
    let again = other.run().unwrap_err();
    assert!(
        again
            .message()
            .starts_with("The static init block at ext::shared:3 failed: RuntimeError:"),
        "{}",
        again.message()
    );
    assert!(
        Function::get_global(&other.marker()).is_err(),
        "a block that failed is not marked"
    );
}

#[test]
fn identity_is_the_crate_version_and_text() {
    fn body() -> Result<()> {
        Ok(())
    }
    let identity = |version, text| StaticInitBlock::__new("ext::ops", 7, version, text, body);
    let text = r#"Function::register_global_typed("ext.clamp", clamp, "Clamp.")?; Ok(())"#;
    let block = identity("0.1.0", text).identity();
    assert!(block.starts_with("ext::ops:7@0.1.0#"), "{block}");

    // The same block in another library, whose compiler may space the tokens otherwise.
    let spaced =
        "Function :: register_global_typed(\"ext.clamp\", clamp, \"Clamp.\") ? ;\n    Ok(())";
    assert_eq!(identity("0.1.0", spaced).identity(), block);
    // A block of another version of the crate, or whose text differs, is another block.
    assert_ne!(identity("0.1.1", text).identity(), block);
    let renamed = text.replace("ext.clamp", "ext.clip");
    assert_ne!(identity("0.1.0", &renamed).identity(), block);
}
