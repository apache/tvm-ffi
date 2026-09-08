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

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, AccessError, ThreadId};

use super::Function;
use crate::{Any, AnyView, Error, Result, RUNTIME_ERROR};

type Callback = dyn Fn(&[AnyView]) -> Result<Any>;

struct LocalCallbacks {
    next_id: usize,
    callbacks: HashMap<usize, Rc<Callback>>,
    release_sender: Sender<usize>,
    released: Receiver<usize>,
}

impl LocalCallbacks {
    fn new() -> Self {
        let (release_sender, released) = mpsc::channel();
        Self {
            next_id: 0,
            callbacks: HashMap::new(),
            release_sender,
            released,
        }
    }
}

thread_local! {
    static CALLBACKS: RefCell<LocalCallbacks> = RefCell::new(LocalCallbacks::new());
}

fn with_callbacks<R>(
    f: impl FnOnce(&mut LocalCallbacks) -> R,
) -> std::result::Result<R, AccessError> {
    let (result, released) = CALLBACKS.try_with(|callbacks| {
        let mut callbacks = callbacks.borrow_mut();
        let mut released = Vec::new();
        while let Ok(id) = callbacks.released.try_recv() {
            if let Some(callback) = callbacks.callbacks.remove(&id) {
                released.push(callback);
            }
        }
        (f(&mut callbacks), released)
    })?;
    // A capture's destructor may create, call, or drop another local function.
    // Run it after releasing the registry borrow, just like the callback itself.
    drop(released);
    Ok(result)
}

// Only this handle crosses the ABI; the callback and its captures stay in TLS.
struct LocalCallbackHandle {
    owner: ThreadId,
    id: usize,
    release_sender: Sender<usize>,
}

impl LocalCallbackHandle {
    fn call(&self, args: &[AnyView]) -> Result<Any> {
        if self.owner != thread::current().id() {
            return Err(Error::new(
                RUNTIME_ERROR,
                "thread-local function called from a different thread",
                "",
            ));
        }
        let callback = with_callbacks(|callbacks| callbacks.callbacks.get(&self.id).cloned())
            .ok()
            .flatten()
            .ok_or_else(|| {
                Error::new(
                    RUNTIME_ERROR,
                    "thread-local function is no longer available on its creating thread",
                    "",
                )
            })?;
        callback(args)
    }
}

impl Drop for LocalCallbackHandle {
    fn drop(&mut self) {
        if self.owner == thread::current().id() {
            // A failed TLS lookup means thread teardown already owns cleanup.
            let _ = with_callbacks(|callbacks| callbacks.callbacks.remove(&self.id));
        } else {
            // If the owner has exited, its TLS has already dropped the captures.
            let _ = self.release_sender.send(self.id);
        }
    }
}

pub(super) fn new<F>(func: F) -> Function
where
    F: Fn(&[AnyView]) -> Result<Any> + 'static,
{
    let callback: Rc<Callback> = Rc::new(func);
    let handle = with_callbacks(|callbacks| {
        let id = callbacks.next_id;
        callbacks.next_id = id.checked_add(1).expect("local callback IDs exhausted");
        callbacks.callbacks.insert(id, callback);
        LocalCallbackHandle {
            owner: thread::current().id(),
            id,
            release_sender: callbacks.release_sender.clone(),
        }
    })
    .expect("cannot create a local function during thread-local teardown");
    Function::from_packed(move |args| handle.call(args))
}
