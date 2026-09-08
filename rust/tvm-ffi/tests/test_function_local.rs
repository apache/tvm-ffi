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

use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use tvm_ffi::{Any, Array, Function, ObjectIdentity, Result};

#[test]
fn local_callbacks_preserve_captures_and_reenter_through_cpp() {
    assert_eq!(unsafe { tvm_ffi_sys::TVMFFITestingDummyTarget() }, 0);
    let array = Array::new(vec![7i64]);
    let captured = array.clone();
    let calls = Rc::new(Cell::new(0));
    let state = calls.clone();
    let function = Function::from_typed_local(move |value: i64| {
        tvm_ffi::ensure!(value >= 0, tvm_ffi::VALUE_ERROR, "negative input");
        state.set(state.get() + 1);
        Ok(captured.clone())
    });
    let retained: Function = tvm_ffi::cached_global_func!("testing.echo")
        .call_tuple((&function,))
        .unwrap()
        .try_into()
        .unwrap();
    drop(function);
    let outer = Function::from_packed_local(move |args| retained.call_packed(args));
    let result: Array<i64> = tvm_ffi::cached_global_func!("testing.apply")
        .call_tuple((&outer, 1i64))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(ObjectIdentity::of(&result), ObjectIdentity::of(&array));
    assert_eq!(calls.get(), 1);
    let error = outer.call_tuple((-1i64,)).err().unwrap();
    assert!(error.to_string().contains("negative input"));
    drop(outer);
    assert_eq!(Rc::strong_count(&calls), 1);
}

#[test]
fn foreign_calls_fail_without_touching_local_state() {
    let calls = Rc::new(Cell::new(0));
    let state = calls.clone();
    let function = Function::from_typed_local(move || {
        state.set(state.get() + 1);
        Ok(())
    });
    thread::scope(|scope| {
        let error = scope
            .spawn(|| {
                tvm_ffi::cached_global_func!("testing.apply")
                    .call_tuple((&function,))
                    .err()
                    .unwrap()
                    .to_string()
            })
            .join()
            .unwrap();
        assert!(error.contains("different thread"), "{error}");
    });
    assert_eq!(calls.get(), 0);
    function.call_tuple(()).unwrap();
    assert_eq!(calls.get(), 1);
}

struct CallOnDrop(Function);

impl Drop for CallOnDrop {
    fn drop(&mut self) {
        let _ = self.0.call_tuple(());
    }
}

#[test]
fn foreign_release_defers_cleanup_and_destructors_can_reenter() {
    let drops = Rc::new(Cell::new(0));
    let state = drops.clone();
    let cleanup = CallOnDrop(Function::from_typed_local(move || {
        state.set(state.get() + 1);
        Ok(())
    }));
    let function = Function::from_typed_local(move || {
        let _keep_capture = &cleanup;
        Ok(())
    });
    thread::spawn(move || {
        // The C++ container releases the last function handle on this thread.
        let container = tvm_ffi::cached_global_func!("ffi.Array")
            .call_tuple((&function,))
            .unwrap();
        drop(function);
        drop(container);
    })
    .join()
    .unwrap();
    assert_eq!(drops.get(), 0);
    drop(Function::from_typed_local(|| Ok(())));
    assert_eq!(drops.get(), 1);
    assert_eq!(Rc::strong_count(&drops), 1);
}

struct OwnerDrop {
    owner: thread::ThreadId,
    drops: Arc<AtomicUsize>,
    on_drop: Function,
    _local: Rc<()>,
}

impl Drop for OwnerDrop {
    fn drop(&mut self) {
        assert_eq!(thread::current().id(), self.owner);
        let error = self.on_drop.call_tuple(()).err().unwrap();
        assert!(error.to_string().contains("no longer available"));
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn owner_exit_releases_captures_even_when_native_code_retains_the_handle() {
    let drops = Arc::new(AtomicUsize::new(0));
    let state = drops.clone();
    let retained = thread::spawn(move || {
        let owner_drop = OwnerDrop {
            owner: thread::current().id(),
            drops: state,
            on_drop: Function::from_typed_local(|| Ok(())),
            _local: Rc::new(()),
        };
        let function = Function::from_packed_local(move |_| -> Result<Any> {
            let _keep_capture = &owner_drop;
            Ok(Any::new())
        });
        tvm_ffi::cached_global_func!("testing.echo")
            .call_tuple((&function,))
            .and_then(Function::try_from)
            .unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let error = retained.call_tuple(()).err().unwrap();
    assert!(error.to_string().contains("different thread"));
    drop(retained);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
}
