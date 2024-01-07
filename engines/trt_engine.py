from collections import OrderedDict
from copy import copy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import os
import math
import glob
import ctypes
from PIL import Image
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine, set_layer_precisions
from polygraphy.backend.trt import util as trt_util
from polygraphy import cuda
import random
from scipy import integrate
import tensorrt as trt
import torch
import requests
from io import BytesIO

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

np.bool = np.bool_

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def device_view(t):
    return cuda.DeviceView(ptr=t.data_ptr(), shape=t.shape, dtype=torch_to_numpy_dtype_dict[t.dtype])

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
for soFilePath in glob.glob("/home/player/trt-hackathon/csrc/**/*.so", recursive=True):
     print(soFilePath)
     ctypes.cdll.LoadLibrary(soFilePath)

def getLayerNormPlugin():
    for c in trt.get_plugin_registry().plugin_creator_list:
        print(c.name)

    if c.name == 'LayerNormalization':
            return c.create_plugin(c.name, trt.PluginFieldCollection([]))
    return None

# getLayerNormPlugin()

class Engine():
    def __init__(
        self,
        engine_path,
    ):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()

    def __del__(self):
        [buf.free() for buf in self.buffers.values() if isinstance(buf, cuda.DeviceArray) ]
        del self.engine
        del self.context
        del self.buffers
        del self.tensors

    def refit(self, onnx_path, onnx_refit_path):
        def convert_int64(arr):
            # TODO: smarter conversion
            if len(arr.shape) == 0:
                return np.int32(arr)
            return arr

        def add_to_map(refit_dict, name, values):
            if name in refit_dict:
                assert refit_dict[name] is None
                if values.dtype == np.int64:
                    values = convert_int64(values)
                refit_dict[name] = values

        print(f"Refitting TensorRT engine with {onnx_refit_path} weights")
        refit_nodes = gs.import_onnx(onnx.load(onnx_refit_path)).toposort().nodes

        # Construct mapping from weight names in refit model -> original model
        name_map = {}
        for n, node in enumerate(gs.import_onnx(onnx.load(onnx_path)).toposort().nodes):
            refit_node = refit_nodes[n]
            assert node.op == refit_node.op
            # Constant nodes in ONNX do not have inputs but have a constant output
            if node.op == "Constant":
                name_map[refit_node.outputs[0].name] = node.outputs[0].name
            # Handle scale and bias weights
            elif node.op == "Conv":
                if node.inputs[1].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTKERNEL"] = node.name+"_TRTKERNEL"
                if node.inputs[2].__class__ == gs.Constant:
                    name_map[refit_node.name+"_TRTBIAS"] = node.name+"_TRTBIAS"
            # For all other nodes: find node inputs that are initializers (gs.Constant)
            else:
                for i, inp in enumerate(node.inputs):
                    if inp.__class__ == gs.Constant:
                        name_map[refit_node.inputs[i].name] = inp.name
        def map_name(name):
            if name in name_map:
                return name_map[name]
            return name

        # Construct refit dictionary
        refit_dict = {}
        refitter = trt.Refitter(self.engine, TRT_LOGGER)
        all_weights = refitter.get_all()
        for layer_name, role in zip(all_weights[0], all_weights[1]):
            # for speciailized roles, use a unique name in the map:
            if role == trt.WeightsRole.KERNEL:
                name = layer_name+"_TRTKERNEL"
            elif role == trt.WeightsRole.BIAS:
                name = layer_name+"_TRTBIAS"
            else:
                name = layer_name

            assert name not in refit_dict, "Found duplicate layer: " + name
            refit_dict[name] = None


        for n in refit_nodes:
            # Constant nodes in ONNX do not have inputs but have a constant output
            if n.op == "Constant":
                name = map_name(n.outputs[0].name)
                print(f"Add Constant {name}\n")
                add_to_map(refit_dict, name, n.outputs[0].values)

            # Handle scale and bias weights
            elif n.op == "Conv":
                if n.inputs[1].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTKERNEL")
                    add_to_map(refit_dict, name, n.inputs[1].values)

                if n.inputs[2].__class__ == gs.Constant:
                    name = map_name(n.name+"_TRTBIAS")
                    add_to_map(refit_dict, name, n.inputs[2].values)

            # For all other nodes: find node inputs that are initializers (AKA gs.Constant)
            else:
                for inp in n.inputs:
                    name = map_name(inp.name)
                    if inp.__class__ == gs.Constant:
                        add_to_map(refit_dict, name, inp.values)

        for layer_name, weights_role in zip(all_weights[0], all_weights[1]):
            if weights_role == trt.WeightsRole.KERNEL:
                custom_name = layer_name+"_TRTKERNEL"
            elif weights_role == trt.WeightsRole.BIAS:
                custom_name = layer_name+"_TRTBIAS"
            else:
                custom_name = layer_name

            # Skip refitting Trilu for now; scalar weights of type int64 value 1 - for clip model
            if layer_name.startswith("onnx::Trilu"):
                continue

            if refit_dict[custom_name] is not None:
                refitter.set_weights(layer_name, weights_role, refit_dict[custom_name])
            else:
                print(f"[W] No refit weights for layer: {layer_name}")

        if not refitter.refit_cuda_engine():
            print("Failed to refit!")
            exit(0)

    def build(self, onnx_path, fp16, input_profile=None, enable_refit=False, enable_preview=True, enable_all_tactics=True, timing_cache=None, workspace_size=0):
        print(f"Building TensorRT engine for {onnx_path}: {self.engine_path}")
        p = Profile()
        if input_profile:
            for name, dims in input_profile.items():
                assert len(dims) == 3
                p.add(name, min=dims[0], opt=dims[1], max=dims[2])

        config_kwargs = {}

        config_kwargs['preview_features'] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
        # config_kwargs['preview_features'] = []
        if enable_preview:
            # Faster dynamic shapes made optional since it increases engine build time.
            config_kwargs['preview_features'].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
        if workspace_size > 0:
            config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
        if not enable_all_tactics:
            config_kwargs['tactic_sources'] = []
        

        builder, network, parser = network_from_onnx_path(onnx_path)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            # print(layer.name, "  ", layer.type, " ", layer.precision)
            if 'InstanceNormalization' in layer.name:
                layer.precision = trt.float32

        engine = engine_from_network(
            (builder, network, parser),
            config=CreateConfig(fp16=fp16,
                refittable=enable_refit,
                profiles=[p],
                load_timing_cache=timing_cache,
                **config_kwargs
            ),
            save_timing_cache=timing_cache
        )


        save_engine(engine, path=self.engine_path)

    def load(self):
        print(f"Loading TensorRT engine: {self.engine_path}")
        self.engine = engine_from_bytes(bytes_from_path(self.engine_path))
        # network = self.engine.network
        # for i in range(network.num_layers):
        #     layer = network.get_layer(i)
        #     print(layer.name, "  ", layer.type, " ", layer.precision)
        #     if 'InstanceNormalization' in layer.name or layer.type == trt.LayerType.NORMALIZATION:
        #         layer.precision = trt.float32

    def activate(self):
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, shape_dict=None, batch_size=1, device='cuda'):
        # for idx in range(trt_util.get_bindings_per_profile(self.engine)):
        for idx in range(self.engine.num_bindings):
            binding = self.engine[idx]
            if shape_dict and binding in shape_dict:
                shape = shape_dict[binding]
            else:
                shape = self.engine.get_binding_shape(binding)

            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            if shape[0] == -1: shape[0] = batch_size
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(idx, shape)
            tensor = torch.zeros(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[binding] = tensor
            self.buffers[binding] = cuda.DeviceView(ptr=tensor.data_ptr(), shape=shape, dtype=dtype)
        # print(self.buffers)

    def infer(self, feed_dict, stream):
        # start_binding, end_binding = trt_util.get_active_profile_bindings(self.context)
        # start_binding = self.engine.num_bindings
        start_binding = 0
        # print(start_binding)
        # print(feed_dict)

        # shallow copy of ordered dict
        device_buffers = copy(self.buffers)
        for name, buf in feed_dict.items():
            assert isinstance(buf, cuda.DeviceView)
            device_buffers[name] = buf
        # bindings = [0] * start_binding + [buf.ptr for buf in device_buffers.values()]
        # print("buffers:", device_buffers)
        # print("bindings:", bindings)
        # noerror = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)
            
        for name, buf in device_buffers.items():
            self.context.set_tensor_address(name, buf.ptr)

        noerror = self.context.execute_async_v3(stream_handle=stream.ptr)
        
        if not noerror:
            raise ValueError(f"ERROR: inference failed.")

        return self.tensors
