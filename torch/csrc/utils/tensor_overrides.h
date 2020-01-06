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

static py::object PyObject_FastGetAttrString(PyObject *obj, char *name);

/*
 * Checks if obj has a __torch_function__ implementation
 *
 * Returns true if an implementation is found and false otherwise
 *
 */

auto check_has_torch_function(PyObject* obj) -> bool;

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

namespace python {

void _implement_torch_function(py::function implementation, py::function public_api, py::iterable relevant_args, py::tuple args, py::dict kwargs);

} // namespace python
} // namespace torch
