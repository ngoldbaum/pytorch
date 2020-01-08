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

py::object PyObject_FastGetAttrString(PyObject *obj, char *name)
{
    PyTypeObject *tp = Py_TYPE(obj);
    PyObject *res = (PyObject *)NULL;

    /* Attribute referenced by (char *)name */
    if (tp->tp_getattr != NULL) {
        res = (*tp->tp_getattr)(obj, name);
        if (res == NULL) {
          PyErr_Clear();
        }
    }
    /* Attribute referenced by (PyObject *)name */
    else if (tp->tp_getattro != NULL) {
        PyObject *w = THPUtils_internString(name);
        if (w == NULL) {
          return py::object();
        }
        res = (*tp->tp_getattro)(obj, w);
        Py_DECREF(w);
        if (res == NULL) {
            PyErr_Clear();
        }
    }
    return py::reinterpret_steal<py::object>(res);
}

// Makes sure that we don't check for __torch_function__ on basic Python types
bool _is_basic_python_type(PyTypeObject *tp)
{
  return (
    /* Basic number types */
    tp == &PyBool_Type ||

    tp == &PyLong_Type ||
    tp == &PyFloat_Type ||
    tp == &PyComplex_Type ||

    /* Basic sequence types */
    tp == &PyList_Type ||
    tp == &PyTuple_Type ||
    tp == &PyDict_Type ||
    tp == &PySet_Type ||
    tp == &PyFrozenSet_Type ||
    tp == &PyUnicode_Type ||
    tp == &PyBytes_Type ||

#if PY_MAJOR_VERSION == 2
    tp == &PyString_Type ||
#endif

    /* other builtins */
    tp == &PySlice_Type ||
    tp == Py_TYPE(Py_None) ||
    tp == Py_TYPE(Py_Ellipsis) ||
    tp == Py_TYPE(Py_NotImplemented) ||

    PyModule_Check(tp) ||
    /* sentinel to swallow trailing || */
    false
  );
}

/*
 * Lookup a special method, following the python approach of looking up
 * on the type object, rather than on the instance itself.
 *
 * Assumes that the special method is a torch-specific one, so does not
 * look at builtin types, nor does it look at a base Tensor.
 *
 * If no special method is found, return NULL, otherwise returns a new
 * reference to the function object
 *
 * In future, could be made more like _Py_LookupSpecial
 */

py::object PyTorch_LookupSpecial(PyObject *obj, char* name)
{
  PyTypeObject *tp = Py_TYPE(obj);
  if (THPVariable_CheckExact(obj)) {
      return py::object();
  }
  if (_is_basic_python_type(tp)) {
    return py::object();
  }
  if(PyObject_HasAttrString(obj, name) == 0){
    return py::object();
  }
  return PyObject_FastGetAttrString((PyObject *)tp, name);
}

auto check_has_torch_function(PyObject* obj) -> bool
{
  py::object method = PyTorch_LookupSpecial(obj, "__torch_function__");
  if(method.ptr() != nullptr){
    return true;
  }
  return false;
}

py::object handle_torch_function_from_overloaded_args(const std::vector<py::handle> &overloaded_args, py::object torch_api_function, const std::string &func_name, PyObject* args, PyObject* kwargs) {
  py::object ret;
  for (auto &arg : overloaded_args) {
    py::object torch_function = PyObject_FastGetAttrString(arg.ptr(), "__torch_function__");
    ret = py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(torch_function.ptr(), torch_api_function.ptr(), args, kwargs, NULL));
    if (ret.ptr() != Py_NotImplemented) {
      // Return the reference to the result. This also covers the case where ret
      // is NULL and __torch_function__ raised an exception, which we throw below
      break;
    }
  }
  if (ret.ptr() == nullptr) {
    // if an exception occurred in a user's implementation of
    // __array_function__, throw it
    throw python_error();
  }
  else if (ret.ptr() == Py_NotImplemented) {
    // all __torch_function__ implementations in overloaded_args
    // returned NotImplemented, so we raise a TypeError.
    std::stringstream ss;
    ss << "no implementation found for 'torch." << func_name
       << "' on types that implement __torch_function__: [";
    for (auto &arg : overloaded_args) {
      ss << arg.ptr()->ob_type->tp_name;
      if (!arg.is(overloaded_args.back())) {
        ss << ", ";
      }
      else {
        ss << "]";
      }
    }
    const std::string& tmp = ss.str();
    PyErr_SetString(PyExc_TypeError, tmp.c_str());
    throw python_error();
  }
  return ret;
}

namespace python {

py::object _implement_torch_function(py::function implementation, py::function public_api, std::string func_name, py::iterable relevant_args, py::tuple args, py::dict kwargs) {
  std::vector<py::handle> overloaded_args;

  for (auto iter = py::iter(relevant_args); iter != py::iterator::sentinel(); ++iter) {
    if (check_has_torch_function((*iter).ptr())) {
      append_overloaded_arg(overloaded_args, (*iter).ptr());
    }
  }

  return handle_torch_function_from_overloaded_args(overloaded_args, public_api, func_name, args.ptr(), kwargs.ptr());
  
}

} // namespace python
} // namespace torch
