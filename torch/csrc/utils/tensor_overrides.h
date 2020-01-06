#pragma once

// Utilities to support overriding PyTorch functionality via
// the __torch_function__ protocol.
//
// TODO expand this header to cross-reference numpy and python
// implementation

#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_strings.h>

namespace torch {

/*
 * Reference: https://github.com/numpy/numpy/blob/f4c497c768e0646df740b647782df463825bfd27/numpy/core/src/common/get_attr_string.h#L42
 *
 * Stripped down version of PyObject_GetAttrString,
 * avoids lookups for None, tuple, and List objects,
 * and doesn't create a PyErr since this code ignores it.
 *
 * This can be much faster then PyObject_GetAttrString where
 * exceptions are not used by caller.
 *
 * 'obj' is the object to search for attribute.
 *
 * 'name' is the attribute to search for.
 *
 * Returns a py::object wrapping the return value. If the attribute lookup failed
 * the value will be NULL.
 *
 */

static py::object PyObject_FastGetAttrString(PyObject *obj, char *name)
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
static bool _is_basic_python_type(PyTypeObject *tp)
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

static py::object PyTorch_LookupSpecial(PyObject *obj, char* name)
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

/*
 * Checks if obj has a __torch_function__ implementation
 *
 * Returns true if an implementation is found and false otherwise
 *
 */

static auto check_has_torch_function(PyObject* obj) -> bool
{
  py::object method = PyTorch_LookupSpecial(obj, "__torch_function__");
  if(method.ptr() != nullptr){
    return true;
  }
  return false;
}

/*
 *  obj has a __torch_function__ implementation and may either be a
 *  subclass of Tensor or a Tensor-like duck type. We may need to
 *  append this object to the overloaded_args vector, which tracks all
 *  of the arguments with distinct __torch_function__ implementations
 *  we've seen so far.
 *
 *  If this is the first argument we've seen with __torch_function__
 *  defined, we unconditionally add obj to the overloaded_args vector.
 *
 *  If we've already seen arguments with __torch_function__ defined,
 *  then we first need to check if obj is the same type as any of the
 *  entries in overloaded_args.  If so, we can ignore obj since we
 *  already have an entry in overloaded_args with the same
 *  __torch_function__ implementation.
 *
 *  If it's a different type, we then need to check if it's a subclass
 *  of one of the types we've already seen. If so, we need to insert an
 *  entry in overloaded_args for this type with higher precedence than
 *  the superclass.
 *
 *  See torch._overrides._get_overloaded_types_and_args for the equivalent
 *  function in the Python __torch_function__ implementation.
 *
 *  The precedence-determining algorithm implemented in this function is
 *  described in NEP-0018:
 *  https://numpy.org/neps/nep-0018-array-function-protocol.html
 *
 *  'overloaded_args' is a reference to a vector of pybind11 handles
 *  that have distinct __torch_function__ implementations, in order of calling
 *  precedence.
 *
 *  'obj' is an object to check for a __torch_function__ implementation
 *
 */

void append_overloaded_arg(std::vector<py::handle> &overloaded_args, PyObject* obj);
  
} // namespace torch
