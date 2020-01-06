#include <torch/csrc/utils/tensor_overrides.h>

namespace torch {

void append_overloaded_arg(std::vector<py::handle> &overloaded_args, PyObject* obj) {
  bool class_not_seen_yet = true;
  for (auto &arg : overloaded_args) {
    if (Py_TYPE(obj) == Py_TYPE(arg.ptr())) {
      // obj is the same type as another parameter we've seen in a prior
      // iteration of the loop over parameters so we already have an entry
      // with the proper __torch_function__ implementation to call, so skip
      // this parameter
      class_not_seen_yet = false;
      break;
    }
  }
  if (class_not_seen_yet) {
    int arg_index = overloaded_args.size();
    for (int j = 0; j < arg_index; j++) {
      if (PyObject_IsInstance(obj, (PyObject*)(Py_TYPE(overloaded_args[j].ptr())))) {
        // obj is a subclass of another object we've seen already so its
        // __torch_function__ should be called first, therefore we
        // insert it into overloaded_args before the superclass
        arg_index = j;
        break;
      }
    }
    // add object to overloaded_args. If it's a subclass of another class
    // we've already seen it will be inserted before the superclass,
    // otherwise it will be inserted at the end of the array
    overloaded_args.insert(overloaded_args.begin() + arg_index, obj);
  }
}

} // namespace torch
