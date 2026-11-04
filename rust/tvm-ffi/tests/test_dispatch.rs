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

use tvm_ffi::{Any, AnyView, Dispatcher, IntoDispatcher, Result};

#[test]
fn into_dispatcher_supports_owned_borrowed_and_callbacks() -> Result<()> {
    type Outcome = Option<Result<Any>>;
    struct Add {
        calls: usize,
    }
    impl<'v> Dispatcher<(AnyView<'v>, i64), Outcome> for &mut Add {
        fn dispatch_tuple(self, (value, offset): (AnyView<'v>, i64)) -> Outcome {
            value.try_as::<i64>().map(|value| {
                self.calls += 1;
                Ok(Any::from(value + offset))
            })
        }
    }

    // An owned dispatcher keeps its exact argument and result signature.
    let mut owned =
        IntoDispatcher::<(AnyView<'_>, i64), Outcome, _>::into_dispatcher(Add { calls: 0 });
    assert_eq!(
        owned
            .dispatch_tuple((AnyView::from(&40_i64), 2))
            .unwrap()?
            .try_as::<i64>(),
        Some(42)
    );

    // A borrowed dispatcher updates the original instance.
    let mut source = Add { calls: 0 };
    {
        let borrowed =
            IntoDispatcher::<(AnyView<'_>, i64), Outcome, _>::into_dispatcher(&mut source);
        assert_eq!(
            borrowed
                .dispatch_tuple((AnyView::from(&40_i64), 3))
                .unwrap()?
                .try_as::<i64>(),
            Some(43)
        );
    }
    assert_eq!(source.calls, 1);

    // Typed misses fall through; matched callbacks receive only their prefix.
    let callbacks = (
        |_: bool| Any::from(-1_i64),
        |value: i64, offset: i64| Any::from(value + offset),
        |_: AnyView<'_>| -> Any { panic!("a matched callback must stop the tuple") },
    );
    let mut tuple =
        IntoDispatcher::<(AnyView<'_>, i64, &str), Outcome, _>::into_dispatcher(callbacks);
    assert_eq!(
        tuple
            .dispatch_tuple((AnyView::from(&40_i64), 4, "unused"))
            .unwrap()?
            .try_as::<i64>(),
        Some(44)
    );
    assert_eq!(
        tuple
            .dispatch_tuple((AnyView::from(&true), 4, "unused"))
            .unwrap()?
            .try_as::<i64>(),
        Some(-1)
    );
    Ok(())
}
