Optional<VisitInterrupt> StructuralVisitFor(StructuralVisitor* visitor, AnyView self) {
    For op = self.cast<For>();
  
    if (auto ret = visitor->Visit(op->min)) return ret;
    if (auto ret = visitor->Visit(op->extent)) return ret;
    if (auto ret = visitor->Visit(op->body)) return ret;
  
    return std::nullopt;
  }

  TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
  
    refl::TypeAttrDef<ForNode>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(&StructuralVisitFor));
  }

auto result = structuralWalk<Add>(
root,
[](Add op, DefRegionKind kind) -> Variant<WalkResult, VisitInterrupt> {
    // called for every Add
    return WalkResult::kAdvance;
},
WalkOrder::kPreOrder);


std::vector<Var> defs;
std::vector<Var> uses;

structuralWalk<Var>(
    root,
    [&](Var var, DefRegionKind kind) -> Variant<WalkResult, VisitInterrupt> {
        if (kind == DefRegionKind::kNone) {
        uses.push_back(var);
        } else {
        defs.push_back(var);
        }
        return WalkResult::kAdvance;
    }, WalkOrder::kPreOrder);


structuralWalk<Add, Mul, Div>(
    root,
    [](Variant<Add, Mul, Div> op, DefRegionKind kind)
        -> Variant<WalkResult, VisitInterrupt> {
        if (op.as<Add>()) {
            // handle Add
        } else if (op.as<Mul>()) {
        // handle Mul
        }
        return WalkResult::kAdvance;
    }, WalkOrder::kPreOrder);


auto interrupt = structuralWalk<Add, Mul, Div, Function>(
    root,
    [&](Variant<Add, Mul, Div, Function> op,
        DefRegionKind kind) -> Variant<WalkResult, VisitInterrupt> {
        if (auto add = op.as<Add>()) {
            // handle Add
        }
        if (auto mul = op.as<Mul>()) {
            // handle Mul
        }
        return WalkResult::kAdvance;
    },
    WalkOrder::kPreOrder
);


Optional<VisitInterrupt> StructuralVisitFunction(StructuralVisitor* visitor, AnyView value) {
    Function func = value.cast<Function>();
  
    if (auto ret = visitor->WithDefRegionKind(DefRegionKind::kDefRecursive, [&]() {
          return visitor->Visit(func->params);
        })) {
      return ret;
    }
  
    return visitor->Visit(func->body);
  }
  
  TVM_FFI_STATIC_INIT_BLOCK() {
    namespace refl = tvm::ffi::reflection;
  
    refl::TypeAttrDef<FunctionObj>().attr(
        refl::type_attr::kStructuralVisit,
        reinterpret_cast<void*>(&StructuralVisitFunction));
  }
  

  StructuralVisitor visitor;
  Optional<VisitInterrupt> interrupt = visitor.Visit(root);
  if (interrupt) {
    Any payload = (*interrupt)->value;
  }
  

  auto interrupt = structuralWalk<Add, Mul, Div, Function>(
    root,
    [&](Variant<Add, Mul, Div, Function> op,
        DefRegionKind kind) -> Variant<WalkResult, VisitInterrupt> {
        if (auto add = op.as<Add>()) {
            // handle Add
        }
        if (auto mul = op.as<Mul>()) {
            // handle Mul
        }
        return WalkResult::kAdvance;
    },
    WalkOrder::kPreOrder
);

