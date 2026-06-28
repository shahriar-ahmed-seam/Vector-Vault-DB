#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "vectorvault/collection.hpp"
#include "vectorvault/engine.hpp"
#include "vectorvault/error.hpp"
#include "vectorvault/index.hpp"
#include "vectorvault/types.hpp"
#include "vectorvault/version.hpp"

namespace py = pybind11;
using namespace vectorvault;

// ===========================================================================
// Python binding for the Vector-Vault-DB core engine.
//
// Exposes the Engine and Collection surfaces, the shared enums, and the
// exception subclass hierarchy. It performs three conversions at the transport
// boundary: Python numeric sequences become float32 vectors, core error
// categories map onto Python exception types, and query results surface as an
// ordered sequence of (id, float) pairs.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// CoreError + exception registry
// ---------------------------------------------------------------------------
//
// The core engine reports failures by value (Status / Result), never by
// throwing across the ABI boundary. The binding turns an error Status into a
// CoreError carrying the category and the "Category: cause" message, throws it,
// and a single registered translator maps the category onto the matching Python
// exception type.
struct CoreError {
    ErrorCategory category;
    std::string   message;  // "Category: cause"
};

// ErrorCategory -> Python exception type (borrowed pointers kept alive by the
// module holding a reference to each exception object).
std::map<ErrorCategory, PyObject*>& exception_map() {
    static std::map<ErrorCategory, PyObject*> m;
    return m;
}

// Builds a "Category: cause" message and throws it as a CoreError.
[[noreturn]] void raise_core(ErrorCategory category, const std::string& cause) {
    throw CoreError{category, std::string{to_string(category)} + ": " + cause};
}

// Throws when `status` is an error, carrying its category and message.
void throw_if_error(const Status& status) {
    if (status.is_error()) {
        throw CoreError{status.category(), status.message()};
    }
}

// ---------------------------------------------------------------------------
// Vector conversion
// ---------------------------------------------------------------------------

// True when `item` is a Python number the binding accepts as a vector
// component (int, float, bool, or a NumPy scalar). Strings, bytes, None,
// containers, and complex numbers are rejected. PyNumber_Check is true exactly
// for objects exposing an integer/float numeric protocol, which covers NumPy
// scalar types while excluding str/bytes/None/sequences.
bool is_numeric(py::handle item) {
    PyObject* p = item.ptr();
    if (PyComplex_Check(p)) {
        return false;
    }
    return PyNumber_Check(p) != 0;
}

// Converts a Python numeric sequence into a float32 vector. Raises a Python
// TypeError when the argument is not a sequence or when it contains a
// non-numeric element, naming the zero-based index of the first such element.
std::vector<float> to_float_vector(py::handle obj) {
    PyObject* p = obj.ptr();
    // A NumPy array, list, or tuple satisfies the sequence protocol; an int,
    // float, dict, or None does not.
    if (!PySequence_Check(p)) {
        throw py::type_error("vector must be a sequence of numbers");
    }
    const Py_ssize_t n = PySequence_Size(p);
    if (n < 0) {
        throw py::error_already_set();
    }

    std::vector<float> out;
    out.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        py::object item = py::reinterpret_steal<py::object>(PySequence_GetItem(p, i));
        if (!item) {
            throw py::error_already_set();
        }
        if (!is_numeric(item)) {
            throw py::type_error("vector element at index " + std::to_string(i) +
                                 " is not a number");
        }
        const double value = PyFloat_AsDouble(item.ptr());
        if (value == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            throw py::type_error("vector element at index " + std::to_string(i) +
                                 " is not a number");
        }
        out.push_back(static_cast<float>(value));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Metadata conversion
// ---------------------------------------------------------------------------

// Converts a Python dict into the Core_Engine Metadata model. Keys must be
// strings; values must be str / int / float / bool. bool is checked before int
// because bool is a subclass of int in Python.
Metadata to_metadata(const py::dict& dict) {
    Metadata meta;
    for (auto entry : dict) {
        if (!PyUnicode_Check(entry.first.ptr())) {
            throw py::type_error("metadata keys must be strings");
        }
        std::string key = entry.first.cast<std::string>();
        py::handle value = entry.second;
        PyObject* vp = value.ptr();
        if (PyBool_Check(vp)) {
            meta[key] = value.cast<bool>();
        } else if (PyLong_Check(vp)) {
            meta[key] = static_cast<std::int64_t>(value.cast<std::int64_t>());
        } else if (PyFloat_Check(vp)) {
            meta[key] = value.cast<double>();
        } else if (PyUnicode_Check(vp)) {
            meta[key] = value.cast<std::string>();
        } else {
            throw py::type_error("metadata value for key '" + key +
                                 "' must be str, int, float, or bool");
        }
    }
    return meta;
}

// Converts Core_Engine Metadata back into a Python dict.
py::dict from_metadata(const Metadata& meta) {
    py::dict dict;
    for (const auto& [key, value] : meta) {
        std::visit([&](auto&& arg) { dict[py::str(key)] = py::cast(arg); }, value);
    }
    return dict;
}

// ---------------------------------------------------------------------------
// String -> enum parsing (the Pythonic surface accepts strings)
// ---------------------------------------------------------------------------

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Maps a metric string onto the DistanceMetric enum. An unknown metric yields
// an InvalidMetric error surfaced as the corresponding Python exception.
DistanceMetric parse_metric(const std::string& metric) {
    const std::string m = to_lower(metric);
    if (m == "euclidean" || m == "l2") {
        return DistanceMetric::Euclidean;
    }
    if (m == "cosine") {
        return DistanceMetric::Cosine;
    }
    if (m == "dot" || m == "dot_product" || m == "dotproduct" || m == "ip") {
        return DistanceMetric::DotProduct;
    }
    raise_core(ErrorCategory::InvalidMetric, "unknown distance metric '" + metric + "'");
}

// Maps an index-type string onto the IndexType enum. An unknown type yields an
// InvalidArgument error surfaced as the corresponding Python exception.
IndexType parse_index_type(const std::string& type) {
    const std::string t = to_lower(type);
    if (t == "hnsw") {
        return IndexType::HNSW;
    }
    if (t == "ivf") {
        return IndexType::IVF;
    }
    if (t == "none") {
        return IndexType::None;
    }
    raise_core(ErrorCategory::InvalidArgument, "unknown index type '" + type + "'");
}

// ---------------------------------------------------------------------------
// Exception hierarchy construction
// ---------------------------------------------------------------------------

// Creates a new Python exception type `vectorvault.<name>` with the given base
// (a single type or a tuple of bases), registers it on the module, and returns
// it.
py::object make_exception(py::module_& m, const char* name, py::object bases) {
    const std::string qualified = std::string("vectorvault.") + name;
    PyObject* exc = PyErr_NewException(qualified.c_str(), bases.ptr(), nullptr);
    if (exc == nullptr) {
        throw py::error_already_set();
    }
    py::object obj = py::reinterpret_steal<py::object>(exc);
    m.attr(name) = obj;
    return obj;
}

}  // namespace

// The native extension module is named `_vectorvault`; the public package
// `vectorvault` re-exports its surface (see src/python/vectorvault/__init__.py).
PYBIND11_MODULE(_vectorvault, m) {
    m.doc() = "Vector-Vault-DB native core engine bindings (pybind11).";
    m.attr("__version__") = vectorvault::version_string();
    m.def("version", &vectorvault::version_string,
          "Return the Core_Engine version string.");

    // ----- exception hierarchy --------------------------------------------
    //
    // VectorVaultError is the common base (Exception). Categories that map onto
    // a Python builtin (KeyError, MemoryError, OSError, FileNotFoundError,
    // ValueError) derive from both VectorVaultError and that builtin so callers
    // can catch either the family base or the familiar builtin.
    py::object exc_base =
        make_exception(m, "VectorVaultError",
                       py::reinterpret_borrow<py::object>(PyExc_Exception));

    auto subclass = [&](const char* name) {
        return make_exception(m, name, py::make_tuple(exc_base));
    };
    auto subclass_mixed = [&](const char* name, PyObject* builtin) {
        return make_exception(
            m, name,
            py::make_tuple(exc_base, py::reinterpret_borrow<py::object>(builtin)));
    };

    py::object name_conflict      = subclass("NameConflictError");
    py::object invalid_name       = subclass_mixed("InvalidNameError", PyExc_ValueError);
    py::object invalid_dim        = subclass("InvalidDimensionalityError");
    py::object invalid_metric     = subclass("InvalidMetricError");
    py::object deletion_failed    = subclass("DeletionFailedError");
    py::object not_found          = subclass_mixed("NotFoundError", PyExc_KeyError);
    py::object dimension_mismatch = subclass("DimensionMismatchError");
    py::object invalid_value      = subclass("InvalidValueError");
    py::object invalid_identifier = subclass("InvalidIdentifierError");
    py::object invalid_parameter  = subclass("InvalidParameterError");
    py::object invalid_argument   = subclass("InvalidArgumentError");
    py::object undefined_distance = subclass("UndefinedDistanceError");
    py::object allocation_failure = subclass_mixed("AllocationFailureError", PyExc_MemoryError);
    py::object write_failure      = subclass_mixed("WriteFailureError", PyExc_OSError);
    py::object snapshot_not_found = subclass_mixed("SnapshotNotFoundError", PyExc_FileNotFoundError);
    py::object unsupported_ver    = subclass("UnsupportedVersionError");
    py::object corruption         = subclass("CorruptionError");

    auto& xmap = exception_map();
    xmap[ErrorCategory::NameConflict]          = name_conflict.ptr();
    xmap[ErrorCategory::InvalidName]           = invalid_name.ptr();
    xmap[ErrorCategory::InvalidDimensionality] = invalid_dim.ptr();
    xmap[ErrorCategory::InvalidMetric]         = invalid_metric.ptr();
    xmap[ErrorCategory::DeletionFailed]        = deletion_failed.ptr();
    xmap[ErrorCategory::NotFound]              = not_found.ptr();
    xmap[ErrorCategory::DimensionMismatch]     = dimension_mismatch.ptr();
    xmap[ErrorCategory::InvalidValue]          = invalid_value.ptr();
    xmap[ErrorCategory::InvalidIdentifier]     = invalid_identifier.ptr();
    xmap[ErrorCategory::InvalidParameter]      = invalid_parameter.ptr();
    xmap[ErrorCategory::InvalidArgument]       = invalid_argument.ptr();
    xmap[ErrorCategory::UndefinedDistance]     = undefined_distance.ptr();
    xmap[ErrorCategory::AllocationFailure]     = allocation_failure.ptr();
    xmap[ErrorCategory::WriteFailure]          = write_failure.ptr();
    xmap[ErrorCategory::SnapshotNotFound]      = snapshot_not_found.ptr();
    xmap[ErrorCategory::UnsupportedVersion]    = unsupported_ver.ptr();
    xmap[ErrorCategory::Corruption]            = corruption.ptr();

    // Single translator: CoreError -> the Python exception type for its
    // category, set with the "Category: cause" message.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const CoreError& e) {
            auto& map = exception_map();
            auto it = map.find(e.category);
            PyObject* type = (it != map.end()) ? it->second : PyExc_RuntimeError;
            PyErr_SetString(type, e.message.c_str());
        }
    });

    // ----- shared enums ----------------------------------------------------
    py::enum_<DistanceMetric>(m, "DistanceMetric")
        .value("Euclidean", DistanceMetric::Euclidean)
        .value("Cosine", DistanceMetric::Cosine)
        .value("DotProduct", DistanceMetric::DotProduct);

    py::enum_<IndexType>(m, "IndexType")
        .value("NONE", IndexType::None)
        .value("HNSW", IndexType::HNSW)
        .value("IVF", IndexType::IVF);

    // ----- CollectionInfo (listing entry) ---------------------------------
    py::class_<CollectionInfo>(m, "CollectionInfo")
        .def_readonly("name", &CollectionInfo::name)
        .def_readonly("dimensionality", &CollectionInfo::dimensionality)
        .def_readonly("metric", &CollectionInfo::metric)
        .def_readonly("record_count", &CollectionInfo::record_count)
        .def("__repr__", [](const CollectionInfo& info) {
            return "<CollectionInfo name='" + info.name + "' dim=" +
                   std::to_string(info.dimensionality) +
                   " count=" + std::to_string(info.record_count) + ">";
        });

    // ----- Collection ------------------------------------------------------
    //
    // Collections are owned by the Engine registry; the binding hands out
    // non-owning references (py::nodelete holder + reference_internal return
    // policy keep the Engine alive while a Collection handle is in use).
    py::class_<Collection, std::unique_ptr<Collection, py::nodelete>>(m, "Collection")
        .def(
            "insert",
            [](Collection& self, const std::string& id, py::handle vector,
               py::object metadata) {
                std::vector<float> v = to_float_vector(vector);
                std::optional<Metadata> meta;
                if (!metadata.is_none()) {
                    if (!py::isinstance<py::dict>(metadata)) {
                        throw py::type_error("metadata must be a dict or None");
                    }
                    meta = to_metadata(metadata.cast<py::dict>());
                }
                throw_if_error(self.insert(
                    id, span<const float>(v.data(), v.size()), std::move(meta)));
            },
            py::arg("id"), py::arg("vector"), py::arg("metadata") = py::none())
        .def(
            "insert_batch",
            [](Collection& self, py::handle records) {
                // Accepts an iterable of records, each given as a tuple
                // mirroring insert(): a 2-tuple (id, vector) or a 3-tuple
                // (id, vector, metadata) where metadata is a dict or None.
                // Every record is converted up front through the same
                // float32/metadata path as insert(); a single core call then
                // applies the batch atomically, so any validation error leaves
                // the collection unchanged.
                if (!PySequence_Check(records.ptr()) &&
                    !PyIter_Check(records.ptr()) &&
                    !PyObject_HasAttrString(records.ptr(), "__iter__")) {
                    throw py::type_error("records must be an iterable of "
                                         "(id, vector) or (id, vector, metadata) "
                                         "tuples");
                }

                // Materialize the inputs first so a malformed record is
                // reported before any core work begins.
                std::vector<RecordInput> inputs;
                py::iterator it = py::iter(records);
                std::size_t index = 0;
                for (; it != py::iterator::sentinel(); ++it, ++index) {
                    py::handle item = *it;
                    // Each record must be a 2- or 3-element sequence; reject
                    // strings and mappings, which satisfy the sequence protocol
                    // but are not valid (id, vector[, metadata]) tuples.
                    if (PyUnicode_Check(item.ptr()) || PyBytes_Check(item.ptr()) ||
                        PyDict_Check(item.ptr()) || !PySequence_Check(item.ptr())) {
                        throw py::type_error(
                            "record at index " + std::to_string(index) +
                            " must be an (id, vector) or (id, vector, metadata) "
                            "tuple");
                    }
                    const Py_ssize_t fields = PySequence_Size(item.ptr());
                    if (fields != 2 && fields != 3) {
                        throw py::type_error(
                            "record at index " + std::to_string(index) +
                            " must have 2 or 3 elements (id, vector[, metadata])");
                    }

                    py::object id_obj = py::reinterpret_steal<py::object>(
                        PySequence_GetItem(item.ptr(), 0));
                    py::object vec_obj = py::reinterpret_steal<py::object>(
                        PySequence_GetItem(item.ptr(), 1));
                    if (!id_obj || !vec_obj) {
                        throw py::error_already_set();
                    }
                    if (!PyUnicode_Check(id_obj.ptr())) {
                        throw py::type_error("record id at index " +
                                             std::to_string(index) +
                                             " must be a string");
                    }

                    RecordInput record;
                    record.id     = id_obj.cast<std::string>();
                    record.vector = to_float_vector(vec_obj);
                    if (fields == 3) {
                        py::object meta_obj = py::reinterpret_steal<py::object>(
                            PySequence_GetItem(item.ptr(), 2));
                        if (!meta_obj) {
                            throw py::error_already_set();
                        }
                        if (!meta_obj.is_none()) {
                            if (!py::isinstance<py::dict>(meta_obj)) {
                                throw py::type_error(
                                    "metadata for record at index " +
                                    std::to_string(index) +
                                    " must be a dict or None");
                            }
                            record.meta = to_metadata(meta_obj.cast<py::dict>());
                        }
                    }
                    inputs.push_back(std::move(record));
                }

                throw_if_error(self.insert_batch(
                    span<const RecordInput>(inputs.data(), inputs.size())));
            },
            py::arg("records"))
        .def(
            "get",
            [](Collection& self, const std::string& id) -> py::object {
                Result<RecordView> result = self.get(id);
                if (result.is_error()) {
                    throw CoreError{result.category(), result.status().message()};
                }
                const RecordView& view = result.value();
                py::list vec;
                for (float f : view.vector) {
                    vec.append(py::float_(static_cast<double>(f)));
                }
                py::object meta = view.meta.has_value()
                                      ? static_cast<py::object>(from_metadata(*view.meta))
                                      : py::none();
                return py::make_tuple(vec, meta);
            },
            py::arg("id"))
        .def(
            "delete",
            [](Collection& self, const std::string& id) {
                throw_if_error(self.remove(id));
            },
            py::arg("id"))
        .def("count", [](const Collection& self) { return self.count(); })
        .def("index_type", [](const Collection& self) { return self.index_type(); })
        .def(
            "build_index",
            [](Collection& self, const std::string& type, const py::kwargs& kwargs) {
                IndexType index_type = parse_index_type(type);
                IndexParams params;  // documented defaults
                if (kwargs.contains("m")) {
                    params.m = kwargs["m"].cast<std::uint32_t>();
                }
                if (kwargs.contains("ef_construction")) {
                    params.ef_construction = kwargs["ef_construction"].cast<std::uint32_t>();
                }
                if (kwargs.contains("nlist")) {
                    params.nlist = kwargs["nlist"].cast<std::uint32_t>();
                }
                if (kwargs.contains("seed")) {
                    params.seed = kwargs["seed"].cast<std::uint64_t>();
                }
                throw_if_error(self.build_index(index_type, params));
            },
            py::arg("type"))
        .def(
            "query",
            [](const Collection& self, py::handle vector, py::object k_obj,
               const py::kwargs& kwargs) -> py::list {
                std::vector<float> v = to_float_vector(vector);
                // k must be an integer. bool is a Python int subclass
                // and is accepted as such by the Core_Engine validation.
                if (!PyLong_Check(k_obj.ptr())) {
                    raise_core(ErrorCategory::InvalidArgument,
                               "k must be an integer");
                }
                long long k = k_obj.cast<long long>();
                if (k < 1) {
                    raise_core(ErrorCategory::InvalidArgument,
                               "k must be an integer >= 1");
                }
                std::uint32_t k32 =
                    (k > static_cast<long long>(UINT32_MAX))
                        ? UINT32_MAX
                        : static_cast<std::uint32_t>(k);
                QueryParams params;
                if (kwargs.contains("ef_search")) {
                    params.ef_search = kwargs["ef_search"].cast<std::uint32_t>();
                }
                if (kwargs.contains("nprobe")) {
                    params.nprobe = kwargs["nprobe"].cast<std::uint32_t>();
                }
                Result<std::vector<Neighbor>> result =
                    self.query(span<const float>(v.data(), v.size()), k32, params);
                if (result.is_error()) {
                    throw CoreError{result.category(), result.status().message()};
                }
                py::list out;
                for (const Neighbor& n : result.value()) {
                    out.append(py::make_tuple(py::str(n.id),
                                              py::float_(static_cast<double>(n.distance))));
                }
                return out;
            },
            py::arg("vector"), py::arg("k"));

    // ----- Engine ----------------------------------------------------------
    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        .def(
            "create_collection",
            [](Engine& self, const std::string& name, std::uint32_t dim,
               const std::string& metric) -> Collection* {
                DistanceMetric m = parse_metric(metric);
                Result<CollectionHandle> result =
                    self.create_collection(name, dim, m);
                if (result.is_error()) {
                    throw CoreError{result.category(), result.status().message()};
                }
                return result.value().get();
            },
            py::arg("name"), py::arg("dim"), py::arg("metric"),
            py::return_value_policy::reference_internal)
        .def(
            "get_collection",
            [](Engine& self, const std::string& name) -> Collection* {
                Result<CollectionHandle> result = self.get_collection(name);
                if (result.is_error()) {
                    throw CoreError{result.category(), result.status().message()};
                }
                return result.value().get();
            },
            py::arg("name"), py::return_value_policy::reference_internal)
        .def("list_collections",
             [](const Engine& self) { return self.list_collections(); })
        .def(
            "delete_collection",
            [](Engine& self, const std::string& name) {
                throw_if_error(self.delete_collection(name));
            },
            py::arg("name"))
        .def(
            "save",
            [](Engine& self, const std::string& name, const std::string& path) {
                throw_if_error(
                    self.save_collection(name, std::filesystem::path(path)));
            },
            py::arg("name"), py::arg("path"))
        .def(
            "load",
            [](Engine& self, const std::string& path) -> Collection* {
                Result<CollectionHandle> result =
                    self.load_collection(std::filesystem::path(path));
                if (result.is_error()) {
                    throw CoreError{result.category(), result.status().message()};
                }
                return result.value().get();
            },
            py::arg("path"), py::return_value_policy::reference_internal);
}
